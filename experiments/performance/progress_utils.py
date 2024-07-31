from dataclasses import dataclass
import multiprocessing as mp
import queue
import signal
from typing import Callable, Generic, Optional, Protocol, TypeAlias, TypeVar, Union, cast


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

MultiprocessingProgressQueue: TypeAlias = "mp.Queue[Optional[int]]"
ThreadingProgressQueue = queue.Queue[Progress[Context]]

_Arg = TypeVar("_Arg", contravariant=True)
class WithProgress(Generic[_Arg], Protocol):
    __name__: str
    __doc__: str
    def __call__(
        self,
        config: _Arg,
        progress: ThreadingProgressQueue[Context],
        ctx: Context = None,
        timeout: Optional[float] = None,
    ) -> None:
        ...

Config = TypeVar("Config")

def to_mp_with_progress(
    func: Callable[[Config, MultiprocessingProgressQueue], None]
) -> WithProgress[Config]:
    """A decorator. Make the function run in a separate process and report the progress."""
    def _wrapped(
        config: Config,
        progress: ThreadingProgressQueue[Context],
        ctx: Context = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Run the function in a separate process and report the progress.
        Args:
            config: The original arguments to the function.
            progress: The queue to which to report the progress.
            ctx: The context. This is passed to the progress queue when the task is done.
            timeout: The timeout for the progress queue.
        """
        mp_ctx = mp.get_context("spawn")
        with mp_ctx.Manager() as manager:
            mp_progress = cast(MultiprocessingProgressQueue, manager.Queue())
            proc = mp_ctx.Process(target=func, args=(config, mp_progress), daemon=True)
            proc.start()
            done_trials = 0
            error = False
            while True:
                try:
                    p = mp_progress.get(timeout=timeout)
                except queue.Empty:
                    error = True
                    proc.terminate()
                    break
                if p is None:
                    # Done
                    break
                done_trials += p
                progress.put(ProgressUpdate(ctx=ctx, trials=p))
            proc.join()
        if error:
            # Set back the progress
            progress.put(ProgressUpdate(ctx=ctx, trials=-done_trials))
        # Notify the main thread that the task is done
        progress.put(ProgressDone(ctx=ctx, error=error))
    _wrapped.__name__ = func.__name__
    _wrapped.__doc__ = f"{_wrapped.__doc__}\n\nOriginal function:\n{func.__doc__}"
    return _wrapped

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
