# -- coding: utf8 --
import errno
import os


def safe_make_dirs(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
