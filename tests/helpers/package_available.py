import platform

import pkg_resources
from lightning.fabric.accelerators import TPUAccelerator


def _package_available(package_name: str) -> bool:
    """環境内でパッケージが利用可能かどうかを確認します。

    :param package_name: 確認するパッケージの名前。

    :return: パッケージが利用可能な場合は`True`。そうでない場合は`False`。
    """
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False


_TPU_AVAILABLE = TPUAccelerator.is_available()

_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_WANDB_AVAILABLE = _package_available("wandb")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")