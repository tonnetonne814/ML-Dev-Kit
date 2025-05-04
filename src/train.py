from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
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
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """モデルをトレーニングします。トレーニング中に得られた最良の重みを使用して、テストセットで追加的に評価することもできます。

    このメソッドはオプションの@task_wrapperデコレータでラップされており、障害発生時の動作を制御します。マルチランや障害情報の保存などに便利です。

    :param cfg: Hydraによって構成されたDictConfig設定。
    :return: メトリクスとすべてのインスタンス化されたオブジェクトを含む辞書のタプル。
    """
    # pytorch、numpy、python.randomの乱数ジェネレータのシードを設定します。
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"データモジュールをインスタンス化しています <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"モデルをインスタンス化しています <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("コールバックをインスタンス化しています...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("ロガーをインスタンス化しています...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"トレーナーをインスタンス化しています <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("ハイパーパラメータをログに記録しています！")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("トレーニングを開始します！")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("テストを開始します！")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("最良のチェックポイントが見つかりませんでした！テスト用に現在の重みを使用します...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"最良のチェックポイントパス: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # トレーニングとテストのメトリクスをマージします
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """トレーニングのメインエントリーポイント。

    :param cfg: Hydraによって構成されたDictConfig設定。
    :return: 最適化されたメトリック値を持つOptional[float]。
    """
    # 追加ユーティリティを適用します
    # (例：cfgにタグが提供されていない場合はタグを要求する、cfg構造を表示するなど)
    extras(cfg)

    # モデルをトレーニングします
    metric_dict, _ = train(cfg)

    # hydraベースのハイパーパラメータ最適化のためにメトリック値を安全に取得します
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # 最適化されたメトリックを返します
    return metric_value

if __name__ == "__main__":
    main()
