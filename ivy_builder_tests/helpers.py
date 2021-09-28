# global
import os
import shutil
import time


def remove_dirs(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()
    tune_present = False
    tune_path = os.path.join(base_dir, 'tune')
    if os.path.isdir(tune_path):
        tune_present = True
    shutil.rmtree(os.path.join(base_dir, 'log'), ignore_errors=True)
    shutil.rmtree(os.path.join(base_dir, 'chkpt'), ignore_errors=True)
    shutil.rmtree(os.path.join(base_dir, 'tune'), ignore_errors=True)
    shutil.rmtree(os.path.join(base_dir, 'ray'), ignore_errors=True)
    log_fpath = os.path.join(base_dir, 'log.log')
    if os.path.isfile(log_fpath):
        os.remove(log_fpath)
    serialized_model_path = os.path.join(base_dir, 'serialized_model')
    if os.path.isdir(serialized_model_path):
        shutil.rmtree(serialized_model_path, ignore_errors=True)
    elif os.path.isfile(serialized_model_path):
        os.remove(serialized_model_path)
    if tune_present:
        time.sleep(1)
        if os.path.isdir(tune_path):
            remove_dirs(base_dir)
