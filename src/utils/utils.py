import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """タスク開始前にオプションのユーティリティを適用します。

    ユーティリティ：
        - Pythonの警告を無視する
        - コマンドラインからタグを設定する
        - Richライブラリを使用した設定の表示

    :param cfg: 設定ツリーを含むDictConfigオブジェクト。
    """
    # `extras`設定がない場合は戻る
    if not cfg.get("extras"):
        log.warning("Extras設定が見つかりません！ <cfg.extras=null>")
        return

    # Pythonの警告を無効にする
    if cfg.extras.get("ignore_warnings"):
        log.info("Pythonの警告を無効にします！ <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # 設定でタグが提供されていない場合、コマンドラインからユーザーにタグの入力を促す
    if cfg.extras.get("enforce_tags"):
        log.info("タグを強制します！ <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # Richライブラリを使用して設定ツリーを表示する
    if cfg.extras.get("print_config"):
        log.info("Richで設定ツリーを表示します！ <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """タスク関数の実行時に失敗の動作を制御するオプションのデコレータ。

    このラッパーは以下のために使用できます：
        - タスク関数が例外を発生させても、ロガーが確実に閉じられるようにする（マルチラン失敗を防ぐ）
        - 例外を`.log`ファイルに保存する
        - 専用のファイルを`logs/`フォルダに作成して実行を失敗としてマークする（後で見つけて再実行できるように）
        - その他（必要に応じて調整）

    例：
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: ラップされるタスク関数。

    :return: ラップされたタスク関数。
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # タスクを実行
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # 例外が発生した場合の処理
        except Exception as ex:
            # 例外を`.log`ファイルに保存
            log.exception("")

            # 一部のハイパーパラメータの組み合わせは無効であるか、メモリ不足エラーを引き起こす可能性があります
            # したがって、Optunaなどのハイパーパラメータ検索プラグインを使用する場合、
            # マルチランの失敗を避けるために以下の例外の発生を無効にすることができます
            raise ex

        # 成功または例外の後に常に実行する処理
        finally:
            # 出力ディレクトリのパスをターミナルに表示
            log.info(f"出力ディレクトリ: {cfg.paths.output_dir}")

            # wandbがインストールされているか確認
            if find_spec("wandb"):
                import wandb

                # 例外が発生してもwandbのランを常に閉じる（マルチランが失敗しないように）
                if wandb.run:
                    log.info("wandbを閉じています！")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """LightningModuleでログに記録されたメトリックの値を安全に取得します。

    :param metric_dict: メトリック値を含む辞書。
    :param metric_name: 提供された場合、取得するメトリックの名前。
    :return: メトリック名が提供された場合、そのメトリックの値。
    """
    if not metric_name:
        log.info("メトリック名がNoneです！メトリック値の取得をスキップします...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"メトリック値が見つかりません！ <metric_name={metric_name}>\n"
            "LightningModuleでログに記録されたメトリック名が正しいことを確認してください！\n"
            "`hparams_search`設定の`optimized_metric`名が正しいことを確認してください！"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"メトリック値を取得しました！ <{metric_name}={metric_value}>")

    return metric_value