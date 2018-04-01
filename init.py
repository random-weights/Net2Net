import os
import shutil
import yaml

"""
1. Will create all necessary directories and files
2. Will also delete any existing files and directories.
"""


def set_workspace():
    hist_dir = "history"
    temp_dir = "temp"
    try:
        shutil.rmtree(hist_dir,ignore_errors=True)
        shutil.rmtree(temp_dir,ignore_errors = True)
    except Exception:
        pass
    finally:
        os.mkdir(hist_dir)
        os.mkdir(temp_dir)


if __name__ == "__main__":
    set_workspace()