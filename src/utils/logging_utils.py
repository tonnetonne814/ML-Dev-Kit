from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Lightningロガーによって保存される設定部分を制御します。

    さらに以下を保存します：
        - モデルのパラメータ数

    :param object_dict: 以下のオブジェクトを含む辞書：
        - `"cfg"`: メイン設定を含むDictConfigオブジェクト。
        - `"model"`: Lightningモデル。
        - `"trainer"`: Lightningトレーナー。
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("ロガーが見つかりません！ハイパーパラメータのログ記録をスキップします...")
        return

    hparams["model"] = cfg["model"]

    # モデルのパラメータ数を保存
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # すべてのロガーにhparamsを送信
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)