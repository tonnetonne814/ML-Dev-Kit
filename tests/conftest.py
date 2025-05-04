"""このファイルは他のテスト用の設定フィクスチャを準備します。"""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """トレーニング用のデフォルトHydra DictConfigを設定するためのpytestフィクスチャ。

    :return: トレーニング用のデフォルトHydra設定を含むDictConfigオブジェクト。
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # すべてのテスト用のデフォルト設定
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """評価用のデフォルトHydra DictConfigを設定するためのpytestフィクスチャ。

    :return: 評価用のデフォルトHydra設定を含むDictConfig。
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # すべてのテスト用のデフォルト設定
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """`cfg_train_global()`フィクスチャの上に構築されたpytestフィクスチャで、一時的なログパスを生成するための
    一時的なログパス`tmp_path`を受け付けます。

    これは`cfg_train`引数を使用する各テストによって呼び出されます。各テストは独自の一時的なログパスを生成します。

    :param cfg_train_global: 変更される入力DictConfigオブジェクト。
    :param tmp_path: 一時的なログパス。

    :return: `tmp_path`に対応する更新された出力およびログディレクトリを持つDictConfig。
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """`cfg_eval_global()`フィクスチャの上に構築されたpytestフィクスチャで、一時的なログパスを生成するための
    一時的なログパス`tmp_path`を受け付けます。

    これは`cfg_eval`引数を使用する各テストによって呼び出されます。各テストは独自の一時的なログパスを生成します。

    :param cfg_train_global: 変更される入力DictConfigオブジェクト。
    :param tmp_path: 一時的なログパス。

    :return: `tmp_path`に対応する更新された出力およびログディレクトリを持つDictConfig。
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()