"""以下から適応:

https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/runif.py
"""

import sys
from typing import Any, Dict, Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution
from pytest import MarkDecorator

from tests.helpers.package_available import (
    _COMET_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _IS_WINDOWS,
    _MLFLOW_AVAILABLE,
    _NEPTUNE_AVAILABLE,
    _SH_AVAILABLE,
    _TPU_AVAILABLE,
    _WANDB_AVAILABLE,
)


class RunIf:
    """テストの条件付きスキップのためのRunIfラッパー。

    `@pytest.mark`と完全に互換性があります。

    例:

    ```python
        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        skip_windows: bool = False,
        sh: bool = False,
        tpu: bool = False,
        fairscale: bool = False,
        deepspeed: bool = False,
        wandb: bool = False,
        neptune: bool = False,
        comet: bool = False,
        mlflow: bool = False,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        """新しい`@RunIf`の`MarkDecorator`デコレータを作成します。

        :param min_gpus: テストを実行するために必要なGPUの最小数。
        :param min_torch: テストを実行するための最小のPyTorchバージョン。
        :param max_torch: テストを実行するための最大のPyTorchバージョン。
        :param min_python: テストを実行するために必要な最小のPythonバージョン。
        :param skip_windows: Windowsプラットフォームでテストをスキップするかどうか。
        :param tpu: TPUが利用可能かどうか。
        :param sh: テストを実行するために`sh`モジュールが必要かどうか。
        :param fairscale: テストを実行するために`fairscale`モジュールが必要かどうか。
        :param deepspeed: テストを実行するために`deepspeed`モジュールが必要かどうか。
        :param wandb: テストを実行するために`wandb`モジュールが必要かどうか。
        :param neptune: テストを実行するために`neptune`モジュールが必要かどうか。
        :param comet: テストを実行するために`comet`モジュールが必要かどうか。
        :param mlflow: テストを実行するために`mlflow`モジュールが必要かどうか。
        :param kwargs: ネイティブの`pytest.mark.skipif`キーワード引数。
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("Windowsでは実行されません")

        if tpu:
            conditions.append(not _TPU_AVAILABLE)
            reasons.append("TPU")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if fairscale:
            conditions.append(not _FAIRSCALE_AVAILABLE)
            reasons.append("fairscale")

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("deepspeed")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        if neptune:
            conditions.append(not _NEPTUNE_AVAILABLE)
            reasons.append("neptune")

        if comet:
            conditions.append(not _COMET_AVAILABLE)
            reasons.append("comet")

        if mlflow:
            conditions.append(not _MLFLOW_AVAILABLE)
            reasons.append("mlflow")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"必要条件: [{' + '.join(reasons)}]",
            **kwargs,
        )