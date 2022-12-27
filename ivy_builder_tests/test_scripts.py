# global
import json
import os
import ivy
import shutil

# local
from scripts.format_dataset_containers import main


# Tests #
# ------#

def test_format_dataset_containers(dev_str):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    orig_cont_dir = os.path.join(this_dir, 'dataset/containers')
    cont_to_format_dir = os.path.join(this_dir, 'dataset/containers_to_format')
    shutil.rmtree(cont_to_format_dir)
    shutil.copytree(orig_cont_dir, cont_to_format_dir)

    # from format file
    cont_format_fpath = os.path.join(this_dir, 'dataset/new_container_format.json')
    main(cont_to_format_dir, cont_format_fpath=cont_format_fpath)
    new_cont_format = ivy.Container.cont_from_disk_as_json(cont_format_fpath)
    new_cont_fnames = os.listdir(cont_to_format_dir)
    for new_cont_fname in new_cont_fnames:
        new_cont_fpath = os.path.join(cont_to_format_dir, new_cont_fname)
        new_cont = ivy.Container.cont_from_disk_as_json(new_cont_fpath)
        assert ivy.Container.cont_identical([new_cont, new_cont_format], check_types=False)
    shutil.rmtree(cont_to_format_dir)
    shutil.copytree(orig_cont_dir, cont_to_format_dir)

    # from format string
    cont_format_as_str = '{"discounts": true, "rewards": true, "step_types": true, "array": true}'
    main(cont_to_format_dir, cont_format_as_str)
    new_cont_format = ivy.Container(json.loads(cont_format_as_str))
    new_cont_fnames = os.listdir(cont_to_format_dir)
    for new_cont_fname in new_cont_fnames:
        new_cont_fpath = os.path.join(cont_to_format_dir, new_cont_fname)
        new_cont = ivy.Container.cont_from_disk_as_json(new_cont_fpath)
        assert ivy.Container.cont_identical([new_cont, new_cont_format], check_types=False)
    shutil.rmtree(cont_to_format_dir)
    shutil.copytree(orig_cont_dir, cont_to_format_dir)
