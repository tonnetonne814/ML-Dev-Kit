#!/bin/bash
# ==============================================================
#  setup_ubuntu_ai_box_en.sh
#  Initial setup script for Ubuntu 22.04/24.04 GPU workstation
#  tested 2025‑05‑03 (Asia/Tokyo)
# ==============================================================

# --- abort if not running under bash ---
if [[ -z "$BASH_VERSION" ]]; then
  echo "This script must be executed with bash." >&2
  exit 1
fi

set -euo pipefail

# ---------- utilities ----------
confirm() {
  # $1: prompt  $2: default (y/n)
  local d=${2:-y} p="[y/N]"
  [[ $d == [Yy] ]] && p="[Y/n]"
  read -rp "$1 $p " a
  a=${a:-$d}
  [[ $a == [Yy] ]]
}

require_root() {
  (( EUID == 0 )) || { echo "Run with sudo/root." >&2; exit 1; }
}

pause() { read -rp "Press Enter to continue…"; }

# ---------- pre‑checks ----------
require_root
echo "=== Starting Ubuntu AI BOX setup ==="
pause

# ---------- 1. Base packages ----------
echo ">>> Installing base packages"
apt-get update && apt-get upgrade -y
apt-get install -y build-essential gcc curl wget gpg sudo iproute2

# ---------- 2. Auto‑mount SSD / HDD ----------
auto_mount() {
  echo ">>> Scanning for ext4 partitions …"
  # Build list of candidate devices (ext4, with UUID, not mounted, not in fstab)
  mapfile -t CANDIDATES < <(
    lsblk -rpo NAME,SIZE,FSTYPE,UUID,MOUNTPOINT | awk '
      $3=="ext4" && $4!="" && $5=="" {print $0}
    ' | while read -r name size fstype uuid mp; do
         # skip if UUID already exists in fstab
         grep -q "$uuid" /etc/fstab || echo "$name $size $uuid"
       done
  )

  if ((${#CANDIDATES[@]}==0)); then
    echo "No unmounted ext4 devices detected – skipping."
    return
  fi

  echo "Available devices:"
  for i in "${!CANDIDATES[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${CANDIDATES[i]}"
  done
  echo "  [0] Done / skip"

  while true; do
    read -rp "Select a device number (0 to finish): " sel
    [[ $sel =~ ^[0-9]+$ ]] || { echo "Not a number."; continue; }
    (( sel==0 )) && break
    (( sel>=1 && sel<=${#CANDIDATES[@]} )) || { echo "Out of range."; continue; }

    IFS=' ' read -r DEV SIZE UUID <<<"${CANDIDATES[sel-1]}"
    echo "Selected: $DEV  ($SIZE, UUID=$UUID)"

    read -rp "Mount point (e.g. /home/$(logname)/SSD): " MP
    [[ -z $MP ]] && { echo "Mount point cannot be empty."; continue; }

    # Make mount point directory
    mkdir -p "$MP"
    chown "$(logname)":"$(logname)" "$MP"

    # Backup and write fstab entry
    cp /etc/fstab /etc/fstab.backup 2>/dev/null || true
    echo "UUID=$UUID $MP ext4 defaults 0 2" >> /etc/fstab
    echo "Added to /etc/fstab."

    # Remove from candidate list so it isn’t offered again
    unset 'CANDIDATES[sel-1]'
    CANDIDATES=("${CANDIDATES[@]}")

    # Offer more selections if any remain
    ((${#CANDIDATES[@]}==0)) && break
    echo "Remaining devices:"
    for i in "${!CANDIDATES[@]}"; do
      printf "  [%d] %s\n" "$((i+1))" "${CANDIDATES[i]}"
    done
    echo "  [0] Done / skip"
  done
}

if confirm "Automatically mount additional SSD/HDD partitions?" y; then
  auto_mount
fi

# ---------- 3. VS Code ----------
if confirm "Install VS Code?" y; then
  echo ">>> Adding VS Code repository"
  install -D -m 644 <(wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor) \
        /etc/apt/keyrings/packages.microsoft.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/packages.microsoft.gpg] \
https://packages.microsoft.com/repos/code stable main" \
    > /etc/apt/sources.list.d/vscode.list
  apt-get update && apt-get install -y code
fi

# ---------- 4. Google Chrome ----------
if confirm "Install Google Chrome?" y; then
  tmp=$(mktemp -d) && \
  wget -q -O "$tmp/chrome.deb" https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
  apt-get install -y "$tmp/chrome.deb" && rm -rf "$tmp"
fi

# ---------- 5. Temperature monitoring ----------
if confirm "Install lm‑sensors and psensor?" y; then
  apt-get install -y lm-sensors psensor
  yes | sensors-detect --auto
fi

# ---------- 6. OpenSSH ----------
apt-get install -y openssh-server

# ---------- 7. Static IP ----------
if confirm "Configure a static IP?" n; then
  echo "Available NICs:"; ip -o -4 addr show | awk '{print "  - "$2" : "$4}'
  read -rp "NIC: " NIC
  read -rp "Static IP (e.g. 192.168.11.50/24): " IP
  read -rp "Gateway (e.g. 192.168.11.1): " GW
  cat >/etc/netplan/99-custom-network.yaml <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    $NIC:
      addresses: [ $IP ]
      routes:
        - to: default
          via: $GW
      nameservers:
        addresses: [ 8.8.8.8, 8.8.4.4 ]
      dhcp4: no
      optional: true
EOF
  netplan apply
fi

# ---------- 8. Docker ----------
if confirm "Install Docker?" y; then
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  chmod 666 /var/run/docker.sock
fi

# ---------- 9. NVIDIA driver ----------
GPU_INSTALL=no
if confirm "Install NVIDIA driver?" y; then
  echo "Choose install method:"
  PS3="> "
  select m in "autoinstall (recommended)" "manual version"; do
    [[ $REPLY == 1 ]] && GPU_INSTALL=auto && break
    [[ $REPLY == 2 ]] && GPU_INSTALL=manual && break
    echo "Invalid."
  done
  if [[ $GPU_INSTALL == auto ]]; then
    ubuntu-drivers autoinstall
  else
    ubuntu-drivers devices
    read -rp "Package (e.g. nvidia-driver-550): " PKG
    apt-get install -y "$PKG"
  fi
fi

# ---------- 10. NVIDIA Container Toolkit ----------
if [[ $GPU_INSTALL != no ]] && confirm "Install NVIDIA Container Toolkit?" y; then
  dist=$(. /etc/os-release; echo ${ID}${VERSION_ID})
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$dist/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-docker2
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  apt-get install -y nvidia-cuda-toolkit
fi

# ---------- 11. Fan control ----------
if [[ $GPU_INSTALL != no ]] && confirm "Enable GPU fan‑speed control?" n; then
  sed -i '/allowed_users/ a needs_root_rights=yes' /etc/X11/Xwrapper.config
fi

# ---------- 12. CUDA smoke test ----------
if [[ $GPU_INSTALL != no ]] && confirm "Run CUDA Docker smoke test?" y; then
  docker run --rm --gpus all nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
    bash -c "nvidia-smi && nvcc -V"
fi

# ---------- 13. Anaconda ----------
if confirm "Install Anaconda?" y; then
  apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
  ANAC_VER="2024.10-1"
  wget -q "https://repo.anaconda.com/archive/Anaconda3-${ANAC_VER}-Linux-x86_64.sh" -O /tmp/ana.sh
  bash /tmp/ana.sh -b -p "/home/$(logname)/anaconda3"
  echo "source /home/$(logname)/anaconda3/bin/activate" >> "/home/$(logname)/.bashrc"
  su - "$(logname)" -c "source ~/anaconda3/bin/activate && conda init && conda config --set auto_activate_base True"
  rm /tmp/ana.sh
fi

# ---------- 14. Hugging Face CLI ----------
if confirm "Install Hugging Face CLI?" y; then
  su - "$(logname)" -c "pip install -U 'huggingface_hub[cli]' && huggingface-cli login"
fi

echo "=== Setup complete! Reboot the system. ==="
