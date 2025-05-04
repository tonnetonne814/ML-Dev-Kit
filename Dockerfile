# CUDA and CuDNN Readyなイメージを選択
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# yes/no の質問をされないように
ENV DEBIAN_FRONTEND="noninteractive"

# ホームを設定
WORKDIR /home/ubuntu/project

# Ubuntu用のライブラリインストール
RUN apt update && apt install -y gcc \
                                 cmake \
                                 libsndfile1 \
                                 dos2unix
RUN apt-get update -y \
    && apt-get install -y \
        make \
        wget \
        curl \
        unzip \
        git \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev

### anaconda ### ref:https://www.eureka-moments-blog.com/entry/2020/02/22/160931#3-AnacondaPython37%E3%82%92%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB
# RUN set -x && \
#     apt update && \
#     apt upgrade -y
# RUN set -x && \
#     apt install -y wget && \
#     apt install -y sudo
# RUN set -x && \
#     wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
#     bash Anaconda3-2023.09-0-Linux-x86_64.sh -b && \
#     rm Anaconda3-2023.09-0-Linux-x86_64.sh
# ENV PATH $PATH:/root/anaconda3/bin
# RUN conda craete -n environment python==3.12.1 && \
#     conda activate environment
################

# uvのインストール ref:https://docs.astral.sh/uv/getting-started/installation/
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# pyenv&poetry のインストール # ref https://blog.8tak4.com/post/158052756945/dockerfile-pyenv
# ENV HOME /home/user
# ENV PYENV_ROOT /home/user/.pyenv
# ### shimsが無いとpythonが通らない https://qiita.com/makuramoto1/items/b5aa08d5fc1ce6af0fb4
# ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
# ### pyenvに必要なものインストール https://github.com/pyenv/pyenv/blob/master/Dockerfile
# RUN apt-get update -y \
#     && apt-get install -y \
#         make \
#         build-essential \
#         libssl-dev \
#         zlib1g-dev \
#         libbz2-dev \
#         libreadline-dev \
#         libsqlite3-dev \
#         wget \
#         curl \
#         llvm \
#         libncurses5-dev \
#         libncursesw5-dev \
#         xz-utils \
#         tk-dev \
#         libffi-dev \
#         liblzma-dev \
#         git
# RUN git clone https://github.com/yyuu/pyenv.git $PYENV_ROOT
# RUN pyenv --version && \
#     pyenv install 3.11.7 && \
#     pyenv global 3.11.7  && \
#     pyenv rehash
# RUN pip install --upgrade pip
# RUN pip install poetry