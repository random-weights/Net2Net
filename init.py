import os
import shutil
import yaml

"""
1. Will create all necessary directories and files
2. Will also delete any existing files and directories.
"""

hist_dir = "history"
try:
    shutil.rmtree(hist_dir,ignore_errors=True)
except:
    pass
finally:
    os.mkdir(hist_dir)

