from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Richライブラリを使用してDictConfigの内容をツリー構造で表示します。

    :param cfg: Hydraによって構成されたDictConfig。
    :param print_order: 設定コンポーネントが表示される順序を決定します。デフォルトは``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``です。
    :param resolve: DictConfigの参照フィールドを解決するかどうか。デフォルトは``False``です。
    :param save_to_file: 設定をHydra出力フォルダにエクスポートするかどうか。デフォルトは``False``です。
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # `print_order`からフィールドをキューに追加
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"フィールド'{field}'が設定に見つかりません。'{field}'の設定表示をスキップします..."
        )

    # その他のすべてのフィールド（`print_order`で指定されていない）をキューに追加
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # キューから設定ツリーを生成
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # 設定ツリーを表示
    rich.print(tree)

    # 設定ツリーをファイルに保存
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """設定にタグが提供されていない場合、コマンドラインからタグを入力するようユーザーに促します。

    :param cfg: Hydraによって構成されたDictConfig。
    :param save_to_file: タグをHydra出力フォルダにエクスポートするかどうか。デフォルトは``False``です。
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("マルチランを開始する前にタグを指定してください！")

        log.warning("設定にタグが提供されていません。ユーザーにタグの入力を促します...")
        tags = Prompt.ask("カンマ区切りのタグリストを入力してください", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"タグ: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)