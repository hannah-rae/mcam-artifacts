
import threading

class Locked(object):
    def __init__(self, x):
        self._x = x
        self._lock = threading.Lock()
    def __enter__(self):
        self._lock.acquire()
        return self._x
    def __exit__(self, e_type, e_value, e_traceback):
        self._lock.release()
