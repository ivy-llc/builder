# global
import os
import shutil


def remove_dirs(base_dir=''):
    shutil.rmtree(os.path.join(base_dir, 'log'), ignore_errors=True)
    serialized_model_path = os.path.join(base_dir, 'serialized_model')
    if os.path.isdir(serialized_model_path):
        shutil.rmtree(serialized_model_path, ignore_errors=True)
    elif os.path.isfile(serialized_model_path):
        os.remove(serialized_model_path)
