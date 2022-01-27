import signal
import contextlib

class TimeoutException(Exception): pass

@contextlib.contextmanager
def time_limited_execution(seconds):
    if seconds>=0:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if seconds>=0:
            signal.alarm(0)