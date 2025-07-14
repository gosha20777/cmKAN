from functools import wraps
from concurrent.futures import Executor


def concurrent(f):
    @wraps(f)
    def _impl(executor: Executor, *args, **kwargs):
        task = executor.submit(f, *args, **kwargs)
        return task

    return _impl
