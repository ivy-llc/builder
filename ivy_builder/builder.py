# global
import os
import ivy
import json
import argparse
import importlib

# local
from ivy_builder.specs import DataLoaderSpec
from ivy_builder.specs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs import NetworkSpec
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.specs.tuner_spec import TunerSpec
from ivy_builder.abstract.tuner import Tuner


# Utility Methods #
# ----------------#

def _import_arg_specified_class_if_present(args_or_spec, class_str):
    if class_str in args_or_spec:
        return load_class_from_str(args_or_spec[class_str])


def load_class_from_str(full_str):
    mod_str = '.'.join(full_str.split('.')[:-1])
    class_str = full_str.split('.')[-1]
    return getattr(importlib.import_module(mod_str), class_str)


def parse_json_to_cont(json_filepath):
    """
    return the data from json file in the form of a python dict
    """
    return_cont = ivy.Container()
    with open(json_filepath) as json_data_file:
        loaded_dict = json.load(json_data_file)
    for k, v in loaded_dict.items():
        if k == 'parents':
            rel_fpaths = v
            for rel_fpath in rel_fpaths:
                if rel_fpath[-5:] == '.json':
                    parent_json_fname = rel_fpath.split('/')[-1]
                else:
                    parent_json_fname = json_filepath.split('/')[-1]
                    rel_fpath = os.path.join(rel_fpath, parent_json_fname)
                rel_fpath = os.path.normpath(rel_fpath)
                fpath = os.path.normpath(os.path.join('/'.join(json_filepath.split('/')[:-1]), rel_fpath))
                fdir = '/'.join(fpath.split('/')[:-1])
                return_cont = ivy.Container.combine(return_cont, json_spec_from_fpath(fdir, parent_json_fname))
    return ivy.Container.combine(return_cont, loaded_dict)


def json_spec_from_fpath(json_spec_path, json_fname, store_duplicates=False):
    base_dir = json_spec_path
    if not os.path.isdir(base_dir):
        raise Exception('base_dir {} does not exist.'.format(base_dir))
    json_spec = ivy.Container()
    while True:
        fpath = os.path.normpath(os.path.join(base_dir, json_fname))
        if os.path.isfile(fpath):
            if store_duplicates:
                parsed_json_cont = ivy.Container(parse_json_to_cont(fpath))
                duplicate_key_chains = list()

                def map_fn(x, kc):
                    if kc in json_spec:
                        duplicate_key_chains.append(kc)
                        return ivy.Container(duplicated={'parent_dir': json_spec[kc], 'this_dir': x})
                    else:
                        return x

                parsed_json_cont = parsed_json_cont.map(map_fn)
                json_spec = ivy.Container.combine(parsed_json_cont,
                                                  json_spec.prune_key_chains(duplicate_key_chains))
            else:
                json_spec = ivy.Container.combine(ivy.Container(parse_json_to_cont(fpath)), json_spec)
        if base_dir.split('/')[-1] == 'json_args':
            return json_spec
        base_dir = os.path.normpath(os.path.join(base_dir, '..'))


def get_json_args(json_spec_path, keys_to_ignore, keychains_to_ignore, keychain_to_show, defaults=False,
                  store_duplicates=False, current_dir_only=False, spec_names=None):
    spec_names = ivy.default(spec_names,
                             [item.split('.json')[0] for item in os.listdir(json_spec_path) if '.json' in item])
    if defaults:
        defaults = '.defaults'
    else:
        defaults = ''
    cont = ivy.Container()
    if current_dir_only:
        for spec_name in spec_names:
            fpath = os.path.join(json_spec_path, spec_name + '.json' + defaults)
            if os.path.isfile(fpath):
                cont[spec_name] = parse_json_to_cont(fpath)
    else:
        for spec_name in spec_names:
            cont[spec_name] = \
                json_spec_from_fpath(json_spec_path, spec_name + '.json' + defaults, store_duplicates)
    for keychain_to_ignore in keychains_to_ignore:
        if keychain_to_ignore in cont:
            cont[keychain_to_ignore] = 'not_shown'
    cont = cont.set_at_keys(dict(zip(keys_to_ignore, ['not_shown']*len(keys_to_ignore))))
    if ivy.exists(keychain_to_show):
        cont = cont[keychain_to_show]
    return cont


def print_json_args(base_dir=None, keys_to_ignore=None, keychains_to_ignore=None):
    if not ivy.exists(base_dir):
        base_dir = os.getcwd()
    ivy.set_framework('numpy')
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--sub_directory', type=str,
                        help='A sub-directory to print the json args for, default is base_dir passed in.')
    parser.add_argument('-dd', '--diff_directory', type=str,
                        help='The directory from which to compare the difference in specifications.')
    parser.add_argument('-kti', '--keys_to_ignore', type=str, default=keys_to_ignore,
                        help='Keys to ignore when printing the specification.')
    parser.add_argument('-kcti', '--keychains_to_ignore', type=str, default=keychains_to_ignore,
                        help='Key-chains to ignore when printing the specification.')
    parser.add_argument('-kcts', '--keychain_to_show', type=str,
                        help='The key-chain to show. Default is None, in which case all key-chains are shown.')
    parser.add_argument('-sn', '--spec_names', type=str,
                        help='The specification names for the json files. Default is ivy_builder defaults of'
                             '[ dataset_dirs | dataset | data_loader| network | trainer |]')
    parser.add_argument('-d', '--show_defaults', action='store_true',
                        help='Whether to show the default json arguments.'
                             'If unset then the current arguments are shown, not the defaut values.')
    parser.add_argument('-c', '--current_dir_only', action='store_true',
                        help='Whether to only show the json arguments for the current directory,'
                             'without searching through parent directories also.')
    parser.add_argument('-sdo', '--show_diff_only', action='store_true',
                        help='Whether to only show the difference between the current directory'
                             'and the diff directory.')
    parser.add_argument('-sso', '--show_same_only', action='store_true',
                        help='Whether to only show the same entries between the current directory'
                             'and the diff directory.')
    parsed_args = parser.parse_args()
    if (parsed_args.show_diff_only or parsed_args.show_same_only) and not parsed_args.diff_directory:
        raise Exception('show_diff_only and show_same_only flags are only applicable if diff_directory is set.')
    if parsed_args.show_diff_only and parsed_args.show_same_only:
        raise Exception('show_diff_only and show_same_only cannot both be set, please choose one to set.')
    if ivy.exists(parsed_args.spec_names):
        spec_names = [kc[1:-1] for kc in ''.join(parsed_args.spec_names[1:-1]).split(', ')]
    else:
        spec_names = None
    if ivy.exists(parsed_args.sub_directory):
        sub_dir = os.path.normpath(os.path.join(base_dir, parsed_args.sub_directory))
    else:
        sub_dir = base_dir
    if ivy.exists(parsed_args.keys_to_ignore):
        keys_to_ignore = [kc[1:-1] for kc in ''.join(parsed_args.keys_to_ignore[1:-1]).split(', ')]
    else:
        keys_to_ignore = list()
    if ivy.exists(parsed_args.keychains_to_ignore):
        keychains_to_ignore = [kc[1:-1] for kc in ''.join(parsed_args.keychains_to_ignore[1:-1]).split(',')]
    else:
        keychains_to_ignore = list()
    these_json_args = get_json_args(
        sub_dir, keys_to_ignore, keychains_to_ignore, parsed_args.keychain_to_show, parsed_args.show_defaults,
        store_duplicates=True, current_dir_only=parsed_args.current_dir_only, spec_names=spec_names)
    if ivy.exists(parsed_args.diff_directory):
        other_sub_dir = os.path.normpath(os.path.join(base_dir, parsed_args.diff_directory))
        if other_sub_dir == sub_dir:
            raise Exception('Invalid diff_directory {} selected, it is the same as the sub_directory {}.'.format(
                other_sub_dir, sub_dir))
        other_json_args = get_json_args(
            other_sub_dir, keys_to_ignore, keychains_to_ignore, parsed_args.keychain_to_show, parsed_args.show_defaults,
            store_duplicates=True, current_dir_only=parsed_args.current_dir_only, spec_names=spec_names)
        diff_keys = 'diff'
        for sub_folder, other_sub_folder in zip(sub_dir.split('/'), other_sub_dir.split('/')):
            if sub_folder != other_sub_folder:
                diff_keys = [sub_folder, other_sub_folder]
                break
        if parsed_args.show_diff_only:
            mode = 'diff_only'
        elif parsed_args.show_same_only:
            mode = 'same_only'
        else:
            mode = 'all'
        diff_json_args = ivy.Container.diff(these_json_args, other_json_args, mode=mode, diff_keys=diff_keys)
        keyword_color_dict = {'duplicated': 'magenta'}
        if isinstance(diff_keys, list):
            diff_keys_dict = dict(zip(diff_keys, ['red'] * 2))
            keyword_color_dict = {**keyword_color_dict, **diff_keys_dict}
        print(ivy.Container(diff_json_args, keyword_color_dict=keyword_color_dict))
    else:
        print(ivy.Container(these_json_args, keyword_color_dict={'duplicated': 'magenta'}))
    ivy.unset_framework()


def save_dict_as_json(dict_to_save, json_filepath):
    """
    save the python dict as a json file at specified filepath
    """
    with open(json_filepath, 'w+') as json_data_file:
        json.dump(dict_to_save, json_data_file, indent=4)


def spec_to_dict(spec):

    def _is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    prev_spec_len = len([x for x in spec.to_iterator()])
    spec = spec.map(lambda x, kc: x.spec if hasattr(x, 'spec') else x)
    new_spec_len = len([x for x in spec.to_iterator()])
    while new_spec_len > prev_spec_len:
        prev_spec_len = new_spec_len
        spec = spec.map(lambda x, kc: x.spec if hasattr(x, 'spec') else x)
        new_spec_len = len([x for x in spec.to_iterator()])
    spec = spec.map(lambda x, kc: x if _is_jsonable(x) else str(x))
    return spec.to_dict()


def _obj_to_class_str(obj_in):
    return str(obj_in.__class__).split("'")[1]


def trainer_to_spec_args_dict(trainer):
    args_dict = dict()
    args_dict['data_loader_class'] = _obj_to_class_str(trainer.spec.data_loader)
    args_dict['network_class'] = _obj_to_class_str(trainer.spec.network)
    args_dict['trainer_class'] = _obj_to_class_str(trainer)
    args_dict['dataset_dirs_args'] = ivy.Container(trainer.spec.data_loader.spec.dataset_spec.dirs.kwargs).to_dict()
    args_dict['dataset_dirs_class'] = _obj_to_class_str(trainer.spec.data_loader.spec.dataset_spec.dirs)
    args_dict['dataset_spec_args'] = ivy.Container(trainer.spec.data_loader.spec.dataset_spec.kwargs).to_dict()
    args_dict['dataset_spec_class'] = _obj_to_class_str(trainer.spec.data_loader.spec.dataset_spec)
    args_dict['data_loader_spec_args'] = ivy.Container(trainer.spec.data_loader.spec.kwargs).to_dict()
    args_dict['data_loader_spec_class'] = _obj_to_class_str(trainer.spec.data_loader.spec)
    args_dict['network_spec_args'] = ivy.Container(trainer.spec.network.spec.kwargs).to_dict()
    args_dict['network_spec_class'] = _obj_to_class_str(trainer.spec.network.spec)
    args_dict['trainer_spec_args'] = ivy.Container(trainer.spec.kwargs).prune_key_chains(
        ['data_loader', 'network']).to_dict()
    args_dict['trainer_spec_class'] = _obj_to_class_str(trainer.spec)
    return args_dict


def command_line_str_to_spec_cont(spec_str):
    """
    save the python dict as a json file at specified filepath
    """
    if spec_str is not None:
        spec_cont = ivy.Container(json.loads(spec_str.replace("'", '"')))
    else:
        spec_cont = ivy.Container()
    all_keys = ['dataset_dirs', 'dataset', 'data_loader', 'network', 'trainer', 'tuner']
    for key in spec_cont.keys():
        if key not in all_keys:
            raise Exception('spec dict keys must all be one of {}, but found {}'.format(all_keys, key))
    for key in all_keys:
        if key not in spec_cont:
            spec_cont[key] = ivy.Container()
    return spec_cont


# Builder Methods #
# ----------------#

def build_dataset_dirs(dataset_dirs_args=None,
                       dataset_dirs_class=None,
                       json_spec_path=None,
                       spec_cont=None,
                       class_priority=False):
    """
    build dataset directories specification
    """

    # define dataset directories specification arguments
    if dataset_dirs_args is None:
        dataset_dirs_args = dict()
    dataset_dirs_args = ivy.Container(dataset_dirs_args)

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'dataset_dirs_args.json')
    else:
        json_spec = ivy.Container()

    # load from spec dict
    this_spec_cont =\
        ivy.Container(spec_cont['dataset_dirs']) if isinstance(spec_cont, dict) and 'dataset_dirs' in spec_cont \
            else ivy.Container()

    # combine args
    dataset_dirs_args = ivy.Container.combine(json_spec, this_spec_cont, dataset_dirs_args)

    # override dataset_dirs_class if specified in dataset_dirs_args
    dataset_dirs_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(dataset_dirs_args, 'dataset_dirs_class'),
        dataset_dirs_class, rev=class_priority),
        DatasetDirs)

    # return dataset directories specification
    return dataset_dirs_class(**dataset_dirs_args)


def build_dataset_spec(dataset_dirs_args=None,
                       dataset_dirs_class=None,
                       dataset_dirs=None,
                       dataset_spec_args=None,
                       dataset_spec_class=None,
                       json_spec_path=None,
                       spec_cont=None,
                       class_priority=False):
    """
    build dataset specification
    """

    # build dataset directories
    dataset_dirs = ivy.default(
        dataset_dirs,
        build_dataset_dirs(
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # define dataset specification arguments
    if dataset_spec_args is None:
        dataset_spec_args = dict()
    dataset_spec_args = ivy.Container(dataset_spec_args)
    dataset_spec_args = ivy.Container.combine(dataset_spec_args, ivy.Container(dirs=dataset_dirs))

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'dataset_args.json')
    else:
        json_spec = ivy.Container()

    # load from spec dict
    this_spec_cont =\
        ivy.Container(spec_cont['dataset']) if isinstance(spec_cont, dict) and 'dataset' in spec_cont \
            else ivy.Container()

    # combine args
    dataset_spec_args = ivy.Container.combine(json_spec, this_spec_cont, dataset_spec_args)

    # override dataset_spec_class if specified in dataset_spec_args
    dataset_spec_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(dataset_spec_args, 'dataset_spec_class'),
        dataset_spec_class, rev=class_priority),
        DatasetSpec)

    # return dataset specification
    return dataset_spec_class(**dataset_spec_args)


def build_network_specification(dataset_dirs_args=None,
                                dataset_dirs_class=None,
                                dataset_dirs=None,
                                dataset_spec_args=None,
                                dataset_spec_class=None,
                                dataset_spec=None,
                                network_spec_args=None,
                                network_spec_class=None,
                                json_spec_path=None,
                                spec_cont=None,
                                class_priority=False):
    """
    build network specification
    """

    # build dataset specification
    dataset_spec = ivy.default(
        dataset_spec,
        build_dataset_spec(
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # define network specification arguments
    if network_spec_args is None:
        network_spec_args = dict()
    network_spec_args = ivy.Container(network_spec_args)
    network_spec_args = ivy.Container.combine(network_spec_args, ivy.Container(dataset_spec=dataset_spec))

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'network_args.json')
    else:
        json_spec = ivy.Container()

    # load from spec dict
    this_spec_cont =\
        ivy.Container(spec_cont['network']) if isinstance(spec_cont, dict) and 'network' in spec_cont \
            else ivy.Container()

    # combine args
    network_spec_args = ivy.Container.combine(json_spec, this_spec_cont, network_spec_args)

    # override network_spec_class if specified in network_spec_args
    network_spec_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(network_spec_args, 'network_spec_class'),
        network_spec_class, rev=class_priority),
        NetworkSpec)

    # return network
    return network_spec_class(**network_spec_args)


def build_network(network_class=None,
                  dataset_dirs_args=None,
                  dataset_dirs_class=None,
                  dataset_dirs=None,
                  dataset_spec_args=None,
                  dataset_spec_class=None,
                  dataset_spec=None,
                  network_spec_args=None,
                  network_spec_class=None,
                  network_spec=None,
                  json_spec_path=None,
                  spec_cont=None,
                  class_priority=False):
    """
    build network
    """

    # build network specification
    network_spec = ivy.default(
        network_spec,
        build_network_specification(
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            dataset_spec=dataset_spec,
            network_spec_args=network_spec_args,
            network_spec_class=network_spec_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # override network_class if specified in network_spec
    network_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(network_spec, 'network_class'),
        network_class, rev=class_priority),
        None)

    # verify network_class exists
    if not ivy.exists(network_class):
        raise Exception('network_class must either be specified in this build_network() method,'
                        'or network_class attribute must be specified in the network_spec instance')

    # network
    return network_class(network_spec)


def build_data_loader_spec(dataset_dirs_args=None,
                           dataset_dirs_class=None,
                           dataset_dirs=None,
                           dataset_spec_args=None,
                           dataset_spec_class=None,
                           dataset_spec=None,
                           data_loader_spec_args=None,
                           data_loader_spec_class=None,
                           json_spec_path=None,
                           spec_cont=None,
                           class_priority=False):
    """
    build data loader specification
    """

    # build dataset specification
    dataset_spec = ivy.default(
        dataset_spec,
        build_dataset_spec(
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # define data loader specification arguments
    if data_loader_spec_args is None:
        data_loader_spec_args = dict()
    data_loader_spec_args = ivy.Container(data_loader_spec_args)
    data_loader_spec_args = ivy.Container.combine(data_loader_spec_args, ivy.Container(dataset_spec=dataset_spec))

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'data_loader_args.json')
    else:
        json_spec = ivy.Container()

    # load from spec dict
    this_spec_cont =\
        ivy.Container(spec_cont['data_loader']) if isinstance(spec_cont, dict) and 'data_loader' in spec_cont \
            else ivy.Container()

    # combine args
    data_loader_spec_args = ivy.Container.combine(json_spec, this_spec_cont, data_loader_spec_args)

    # override data_loader_spec_class if specified in data_loader_spec_args
    data_loader_spec_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(data_loader_spec_args, 'data_loader_spec_class'),
        data_loader_spec_class, rev=class_priority),
        DataLoaderSpec)

    # return data loader
    return data_loader_spec_class(**data_loader_spec_args)


def build_data_loader(data_loader_class=None,
                      dataset_dirs_args=None,
                      dataset_dirs_class=None,
                      dataset_dirs=None,
                      dataset_spec_args=None,
                      dataset_spec_class=None,
                      dataset_spec=None,
                      data_loader_spec_args=None,
                      data_loader_spec_class=None,
                      data_loader_spec=None,
                      json_spec_path=None,
                      spec_cont=None,
                      class_priority=False):
    """
    build data loader
    """

    # build data loader specification
    data_loader_spec = ivy.default(
        data_loader_spec,
        build_data_loader_spec(
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            dataset_spec=dataset_spec,
            data_loader_spec_args=data_loader_spec_args,
            data_loader_spec_class=data_loader_spec_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # override data_loader_class if specified in data_loader_spec
    data_loader_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(data_loader_spec, 'data_loader_class'),
        data_loader_class, rev=class_priority),
        None)

    # verify data_loader_class exists
    if not ivy.exists(data_loader_class):
        raise Exception('data_loader_class must either be specified in this build_data_loader() method,'
                        'or data_loader_class attribute must be specified in the data_loader_spec instance')

    # return data loader
    return data_loader_class(data_loader_spec)


def build_trainer_spec(data_loader_class=None,
                       network_class=None,
                       dataset_dirs_args=None,
                       dataset_dirs_class=None,
                       dataset_dirs=None,
                       dataset_spec_args=None,
                       dataset_spec_class=None,
                       dataset_spec=None,
                       data_loader_spec_args=None,
                       data_loader_spec_class=None,
                       data_loader_spec=None,
                       data_loader=None,
                       network_spec_args=None,
                       network_spec_class=None,
                       network_spec=None,
                       network=None,
                       trainer_spec_args=None,
                       trainer_spec_class=None,
                       json_spec_path=None,
                       spec_cont=None,
                       class_priority=False):
    """
    build trainer specification
    """

    # build data loader
    data_loader = ivy.default(
        data_loader,
        build_data_loader(
            data_loader_class=data_loader_class,
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            dataset_spec=dataset_spec,
            data_loader_spec_args=data_loader_spec_args,
            data_loader_spec_class=data_loader_spec_class,
            data_loader_spec=data_loader_spec,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # build network
    network = ivy.default(
        network,
        build_network(
            network_class=network_class,
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            network_spec_args=network_spec_args,
            network_spec_class=network_spec_class,
            network_spec=network_spec,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # define trainer specification arguments
    if trainer_spec_args is None:
        trainer_spec_args = dict()
    trainer_spec_args = ivy.Container(trainer_spec_args)
    trainer_spec_args = ivy.Container.combine(trainer_spec_args,
                                              ivy.Container(data_loader=data_loader,
                                                            network=network))

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'trainer_args.json')
    else:
        json_spec = ivy.Container()

    # load from spec dict
    this_spec_cont =\
        ivy.Container(spec_cont['trainer']) if isinstance(spec_cont, dict) and 'trainer' in spec_cont \
            else ivy.Container()

    # combine args
    trainer_spec_args = ivy.Container.combine(json_spec, this_spec_cont, trainer_spec_args)

    # override trainer_spec_class if specified in trainer_spec_args
    trainer_spec_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(trainer_spec_args, 'trainer_spec_class'),
        trainer_spec_class, rev=class_priority),
        TrainerSpec)

    # return trainer specification
    return trainer_spec_class(**trainer_spec_args)


def build_trainer(data_loader_class=None,
                  network_class=None,
                  trainer_class=None,
                  dataset_dirs_args=None,
                  dataset_dirs_class=None,
                  dataset_dirs=None,
                  dataset_spec_args=None,
                  dataset_spec_class=None,
                  dataset_spec=None,
                  data_loader_spec_args=None,
                  data_loader_spec_class=None,
                  data_loader_spec=None,
                  data_loader=None,
                  network_spec_args=None,
                  network_spec_class=None,
                  network_spec=None,
                  network=None,
                  trainer_spec_args=None,
                  trainer_spec_class=None,
                  trainer_spec=None,
                  json_spec_path=None,
                  spec_cont=None,
                  class_priority=False):
    """
    build trainer
    """

    # build trainer spec
    trainer_spec = ivy.default(
        trainer_spec,
        build_trainer_spec(
            data_loader_class=data_loader_class,
            network_class=network_class,
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            dataset_spec=dataset_spec,
            data_loader_spec_args=data_loader_spec_args,
            data_loader_spec_class=data_loader_spec_class,
            data_loader_spec=data_loader_spec,
            data_loader=data_loader,
            network_spec_args=network_spec_args,
            network_spec_class=network_spec_class,
            network_spec=network_spec,
            network=network,
            trainer_spec_args=trainer_spec_args,
            trainer_spec_class=trainer_spec_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # override trainer_class if specified in trainer_spec
    trainer_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(trainer_spec, 'trainer_class'),
        trainer_class, rev=class_priority),
        None)

    # verify trainer_class exists
    if not ivy.exists(trainer_class):
        raise Exception('trainer_class must either be specified in this build_trainer() method,'
                        'or trainer_class attribute must be specified in the trainer_spec instance')

    # return trainer
    return trainer_class(trainer_spec)


def build_tuner_spec(data_loader_class=None,
                     network_class=None,
                     trainer_class=None,
                     dataset_dirs_args=None,
                     dataset_dirs_class=None,
                     dataset_dirs=None,
                     dataset_spec_args=None,
                     dataset_spec_class=None,
                     dataset_spec=None,
                     data_loader_spec_args=None,
                     data_loader_spec_class=None,
                     data_loader_spec=None,
                     data_loader=None,
                     network_spec_args=None,
                     network_spec_class=None,
                     network_spec=None,
                     network=None,
                     trainer_spec_args=None,
                     trainer_spec_class=None,
                     trainer_spec=None,
                     trainer=None,
                     tuner_spec_args=None,
                     tuner_spec_class=None,
                     json_spec_path=None,
                     spec_cont=None,
                     class_priority=False):
    """
    build tuner specification
    """

    # define dataset directories specification arguments
    if tuner_spec_args is None:
        tuner_spec_args = dict()
    tuner_spec_args = ivy.Container(tuner_spec_args)

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'tuner_args.json')
    else:
        json_spec = ivy.Container()

    # load from spec dict
    this_spec_cont =\
        ivy.Container(spec_cont['tuner']) if isinstance(spec_cont, dict) and 'tuner' in spec_cont else ivy.Container()

    # combine args
    tuner_spec_args = ivy.Container.combine(json_spec, this_spec_cont, tuner_spec_args)

    # override tuner_spec_class if specified in tuner_spec_args
    tuner_spec_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(tuner_spec_args, 'tuner_spec_class'),
        tuner_spec_class, rev=class_priority),
        TunerSpec)

    # set framework
    ivy.set_framework(tuner_spec_class(None, **tuner_spec_args).framework)

    # build trainer
    trainer = ivy.default(
        trainer,
        build_trainer(
            data_loader_class=data_loader_class,
            network_class=network_class,
            trainer_class=trainer_class,
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            dataset_spec=dataset_spec,
            data_loader_spec_args=data_loader_spec_args,
            data_loader_spec_class=data_loader_spec_class,
            data_loader_spec=data_loader_spec,
            data_loader=data_loader,
            network_spec_args=network_spec_args,
            network_spec_class=network_spec_class,
            network_spec=network_spec,
            network=network,
            trainer_spec_args=trainer_spec_args,
            trainer_spec_class=trainer_spec_class,
            trainer_spec=trainer_spec,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # return tuner specification
    return tuner_spec_class(trainer,
                            **tuner_spec_args)


def build_tuner(data_loader_class=None,
                network_class=None,
                trainer_class=None,
                dataset_dirs_args=None,
                dataset_dirs_class=None,
                dataset_dirs=None,
                dataset_spec_args=None,
                dataset_spec_class=None,
                dataset_spec=None,
                data_loader_spec_args=None,
                data_loader_spec_class=None,
                data_loader_spec=None,
                data_loader=None,
                network_spec_args=None,
                network_spec_class=None,
                network_spec=None,
                network=None,
                trainer_spec_args=None,
                trainer_spec_class=None,
                trainer_spec=None,
                trainer=None,
                tuner_spec_args=None,
                tuner_spec_class=None,
                tuner_spec=None,
                tuner_class=None,
                json_spec_path=None,
                spec_cont=None,
                class_priority=False):
    """
    build tuner
    """

    # build tuner spec
    tuner_spec = ivy.default(
        tuner_spec,
        build_tuner_spec(
            data_loader_class=data_loader_class,
            network_class=network_class,
            trainer_class=trainer_class,
            dataset_dirs_args=dataset_dirs_args,
            dataset_dirs_class=dataset_dirs_class,
            dataset_dirs=dataset_dirs,
            dataset_spec_args=dataset_spec_args,
            dataset_spec_class=dataset_spec_class,
            dataset_spec=dataset_spec,
            data_loader_spec_args=data_loader_spec_args,
            data_loader_spec_class=data_loader_spec_class,
            data_loader_spec=data_loader_spec,
            data_loader=data_loader,
            network_spec_args=network_spec_args,
            network_spec_class=network_spec_class,
            network_spec=network_spec,
            network=network,
            trainer_spec_args=trainer_spec_args,
            trainer_spec_class=trainer_spec_class,
            trainer_spec=trainer_spec,
            trainer=trainer,
            tuner_spec_args=tuner_spec_args,
            tuner_spec_class=tuner_spec_class,
            json_spec_path=json_spec_path,
            spec_cont=spec_cont))

    # override tuner_class if specified in tuner_spec_args
    tuner_class = ivy.default(ivy.default(
        _import_arg_specified_class_if_present(tuner_spec, 'tuner_class'),
        tuner_class, rev=class_priority),
        Tuner)

    # return tuner
    return tuner_class(
        data_loader_class=data_loader_class,
        network_class=network_class,
        trainer_class=trainer_class,
        dataset_dirs_args=dataset_dirs_args,
        dataset_dirs_class=dataset_dirs_class,
        dataset_dirs=dataset_dirs,
        dataset_spec_args=dataset_spec_args,
        dataset_spec_class=dataset_spec_class,
        dataset_spec=dataset_spec,
        data_loader_spec_args=data_loader_spec_args,
        data_loader_spec_class=data_loader_spec_class,
        data_loader_spec=data_loader_spec,
        data_loader=data_loader,
        network_spec_args=network_spec_args,
        network_spec_class=network_spec_class,
        network_spec=network_spec,
        network=network,
        trainer_spec_args=trainer_spec_args,
        trainer_spec_class=trainer_spec_class,
        trainer_spec=trainer_spec,
        trainer=trainer,
        tuner_spec_args=tuner_spec_args,
        tuner_spec_class=tuner_spec_class,
        tuner_spec=tuner_spec,
        json_spec_path=json_spec_path,
        spec_cont=spec_cont)
