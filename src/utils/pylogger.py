import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """マルチGPU対応のPythonコマンドラインロガー。"""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """プロセスのランクがログメッセージの先頭に付加された状態で、すべてのプロセスでログを記録する
        マルチGPU対応のPythonコマンドラインロガーを初期化します。

        :param name: ロガーの名前。デフォルトは``__name__``。
        :param rank_zero_only: すべてのログをランクゼロのプロセスでのみ発生するように強制するかどうか。デフォルトは`False`。
        :param extra: （オプション）コンテキスト情報を提供する辞書のようなオブジェクト。`logging.LoggerAdapter`を参照。
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """ログが記録されるプロセスのランクをメッセージの先頭に付加した後、
        基礎となるロガーにログ呼び出しを委譲します。`'rank'`が提供されている場合、
        そのランク/プロセスでのみログが発生します。

        :param level: ログを記録するレベル。詳細については`logging.__init__.py`を参照してください。
        :param msg: ログに記録するメッセージ。
        :param rank: ログを記録するランク。
        :param args: 基礎となるロギング関数に渡す追加の引数。
        :param kwargs: 基礎となるロギング関数に渡す追加のキーワード引数。
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("`rank_zero_only.rank`は使用前に設定する必要があります")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)