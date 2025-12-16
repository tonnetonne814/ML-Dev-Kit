from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# setup_rootの上記は以下と同等です:
# - プロジェクトのルートディレクトリをPYTHONPATHに追加する
#       (ユーザーにプロジェクトをパッケージとしてインストールさせる必要がない)
#       (ローカルモジュールをインポートする前に必要 例: `from src import utils`)
# - PROJECT_ROOT環境変数を設定する
#       ("configs/paths/default.yaml"内のパスのベースとして使用される)
#       (これによりコードを実行する場所に関係なく、すべてのファイルパスが同じになる)
# - ルートディレクトリの".env"から環境変数を読み込む
#
# 以下の場合は削除できます:
# 1. プロジェクトをパッケージとしてインストールするか、エントリーファイルをプロジェクトのルートディレクトリに移動する
# 2. "configs/paths/default.yaml"内の`root_dir`を"."に設定する
#
# 詳細情報: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

# このプロジェクトからのインポートは、必ずrootutils.setup_rootの実行後に行う必要がある
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """与えられたチェックポイントをデータモジュールのテストセットで評価します。

    このメソッドはオプションの@task_wrapperデコレータでラップされており、障害発生時の動作を制御します。
    マルチランや障害情報の保存などに便利です。

    :param cfg: Hydraによって構成されたDictConfig設定。
    :return: メトリクスとすべてのインスタンス化されたオブジェクトを含む辞書のタプル。
    """
    assert cfg.ckpt_path

    log.info(f"データモジュールをインスタンス化しています <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"モデルをインスタンス化しています <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("ロガーをインスタンス化しています...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"トレーナーをインスタンス化しています <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("ハイパーパラメータをログに記録しています！")
        log_hyperparameters(object_dict)

    log.info("テストを開始します！")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)

    # 予測にはtrainer.predict(...)を使用します
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """評価のメインエントリーポイント。

    :param cfg: Hydraによって構成されたDictConfig設定。
    """
    # 追加ユーティリティを適用します
    # (例：cfgにタグが提供されていない場合はタグを要求する、cfg構造を表示するなど)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()