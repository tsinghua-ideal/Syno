from dataclasses import dataclass
import multiprocessing as mp
import signal
from typing import Callable, Generic, TypeAlias, TypeVar, Union


Context = TypeVar("Context")
@dataclass
class ProgressUpdate(Generic[Context]):
    ctx: Context
    trials: int
@dataclass
class ProgressDone(Generic[Context]):
    ctx: Context
    error: bool
Progress = Union[ProgressUpdate[Context], ProgressDone[Context]]

ProgressQueue: TypeAlias = "mp.Queue[Progress[Context]]"

def inherit_process_authkey() -> Callable[[], None]:
    authkey = bytes(mp.current_process().authkey)
    def _set_authkey() -> None:
        mp.current_process().authkey = authkey
    return _set_authkey

def ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def inherit_process_authkey_and_ignore_sigint() -> Callable[[], None]:
    _inherit_authkey = inherit_process_authkey()
    def both() -> None:
        _inherit_authkey()
        ignore_sigint()
    return both
