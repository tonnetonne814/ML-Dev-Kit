from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
   import sh


def run_sh_command(command: List[str]) -> None:
   """`pytest`と`sh`パッケージを使用してシェルコマンドを実行するためのデフォルトメソッド。

   :param command: 文字列としてのシェルコマンドのリスト。
   """
   msg = None
   try:
       sh.python(command)
   except sh.ErrorReturnCode as e:
       msg = e.stderr.decode()
   if msg:
       pytest.fail(msg=msg)