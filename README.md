# ML-Dev-Kit

このリポジトリは、[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/main?tab=readme-ov-file#hyperparameter-search)をベースにした、機械学習モデル開発環境を作成するテンプレートリポジトリです。

主に以下の機能が実装されています。
- Dockerによるアプリケーションと実行環境の統一化
- uvによるpythonバージョン管理と、pipより高速なパッケージ管理
- Hydraによるハイパーパラメータ管理
- Optunaによるハイパーパラメータ探索
- Pytorch Lightningによる学習/推論コードの構造化、及び学習テクニック実装の簡略化（以下例）
    - 分散トレーニング(DDP)、DDPシミュレーション(デバッグ用)
    - 勾配積み重ね
    - 勾配クリッピング
    - バッチサイズのオートスケーリング

## 開発環境構築
### Windowsネイティブ開発環境構築
- [nvidia-driver](https://www.nvidia.com/ja-jp/drivers/)の最新版をインストール。
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads)をインストールし、パスを通す。
- [CuDNN 8.9.7 for CUDA 12.x](https://developer.nvidia.com/rdp/cudnn-archive)をダウンロードし、CUDA Toolkitフォルダへ上書き。
- [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)をインストール。

### Ubuntuネイティブ開発環境構築
```bash
bash .devcontainer/ubuntu_machine_setup.sh
```
- 手動で行うならば[この記事参照](https://qiita.com/tf63/items/0c6da72fe749319423b4)

### Docker上での開発環境構築
- どちらのOSでもnvidia-driverをインストール。
- Windowsならば[DockerDesktop](https://www.docker.com/ja-jp/products/docker-desktop/)をインストール。もしくは[WSL上にDockerとnvidia-container-toolkit](https://zenn.dev/k_ts_ngo/articles/87a5ee010a089ahttps://zenn.dev/k_ts_ngo/articles/87a5ee010a089a)をインストール。
- Ubuntuならば[Docker](https://docs.docker.com/engine/install/ubuntu/)と[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)のインストール。
- docker-compose.yamlのvolumesのパスを編集
上記を行った後、以下のコマンドを実行する。
```
docker-compose build --no-cache # Dockerイメージの作成
docker-compose up -d            # Dockerコンテナの作成
docker exec -it mldev /bin/bash # コンテナ内へ
```

## .venv環境作成
```
uv sync --extra cu128     # GPU環境
uv sync --extra cpu       # CPU環境
source .venv/bin/activate # [Ubuntu/WSL]  venv有効化
.venv\Scripts\activate    # [Windows]     venv有効化
```

## VSCode\Cursorの拡張機能の追加
CursorとVSCode両方入っている場合は、どちらかを選択してください。
```
# [Ubuntu/WSL]
dos2unix .devcontainer/install_editor_extention.sh
bash .devcontainer/install_editor_extention.sh

# [Windows]
.devcontainer/install_editor_extention.bat
```
追加されるExtensionは以下です。
```
  ms-python.python                # python基本拡張機能
  ms-python.vscode-pylance        # python基本拡張機能
  ms-python.debugpy               # python基本拡張機能
  ms-python.black-formatter       # python基本拡張機能
  saoudrizwan.claude-dev          # AI コード生成
  GitHub.copilot                  # AI コード生成
  GitHub.copilot-chat             # AI コード生成
  usernamehw.errorlens            # エラー可視化
  ms-pyright.pyright              # Pythonコード向け静的型チェッカー
  charliermarsh.ruff              # コードの品質保持
  shardulm94.trailing-spaces      # 無駄なスペースの可視化
  mhutchie.git-graph              # gitコミットの可視化
  mosapride.zenkaku               # 全角スペースの可視化
  kevinrose.vsc-python-indent     # インデント自動調整
  gruntfuggly.todo-tree           # TODO管理
  aaron-bond.better-comments      # コメントの着色
  njpwerner.autodocstring         # docstring テンプレートを生成
```

## uvを用いたpythonパッケージ管理コマンド
```
uv add "packageName"         # pyproject.tomlに追記してパッケージ追加
uv pip install "packageName" # pyproject.tomlに追記せずにパッケージ追加
uv list                      # インストール済み／利用可能な Python バージョン一覧を表示
uv install 3.xx.yy           # Python 3.xx.yy をインストール
uv use 3.xx.yy               # プロジェクトで使用する Python バージョンを 3.xx.yy に切り替え
```

## cuda周りのチェック
```python
python tests\check_cuda.py
```

## 学習処理 / 評価処理 の開始
```
python src/train.py # 学習
python src/eval.py  # 評価(eval.yamlにcheckpointのpathを追加する必要あり)

tensorboard --logdir logs # 学習/評価ログの確認
```

## その他
```
### 処理を投げっぱなすためのコマンド
# 開始コマンド
nohup python path/to/pythonFile.py > nohupLogs.out
# 停止コマンド
ps -ef | path/to/pythonFile.py # pid表示
kill -9 {pid} # 停止対象のpidを入力
```
```
# GPUのvram使用率や温度などをターミナルに表示
nvidia-smi --query-gpu=timestamp,name,memory.used,temperature.gpu,fan.speed,utilization.gpu --format=csv -l 1
```
```
# WSLで使用するメモリサイズを変更する。
https://qiita.com/suzuk12345/items/b0f6bf0cfb09dae031ae
```