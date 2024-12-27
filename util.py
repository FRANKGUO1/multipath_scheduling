import ctypes
import os
import sys
from contextlib import contextmanager


def c_str(py_str):
    return ctypes.c_char_p(bytes(py_str, 'utf-8')) # 将python中的字符串格式转化为c中的字符串格式


def c_int(py_int):
    return ctypes.c_int(py_int) # 同理，将python中的整型转化为c的整型


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    功能是将输出处理到指定文件中，默认是丢弃。
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)
