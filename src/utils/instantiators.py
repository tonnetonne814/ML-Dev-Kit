from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """設定からコールバックをインスタンス化します。

    :param callbacks_cfg: コールバック設定を含むDictConfigオブジェクト。
    :return: インスタンス化されたコールバックのリスト。
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("コールバック設定が見つかりません！スキップします...")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("コールバック設定はDictConfigでなければなりません！")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"コールバックをインスタンス化しています <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """設定からロガーをインスタンス化します。

    :param logger_cfg: ロガー設定を含むDictConfigオブジェクト。
    :return: インスタンス化されたロガーのリスト。
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("ロガー設定が見つかりません！スキップします...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("ロガー設定はDictConfigでなければなりません！")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"ロガーをインスタンス化しています <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger