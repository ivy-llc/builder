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


def _import_arg_specified_class_if_present(args_or_spec, class_str):
    if class_str in args_or_spec:
        mod_str = '.'.join(args_or_spec[class_str].split('.')[:-1])
        class_str = args_or_spec[class_str].split('.')[-1]
        loaded_class = getattr(importlib.import_module(mod_str), class_str)
        return loaded_class
    return


def json_spec_from_fpath(json_spec_path, json_fname, store_duplicates=False):
    base_dir = json_spec_path
    if not os.path.isdir(base_dir):
        raise Exception('base_dir {} does not exist.'.format(base_dir))
    json_spec = dict()
    while True:
        fpath = os.path.abspath(os.path.join(base_dir, json_fname))
        if os.path.isfile(fpath):
            if store_duplicates:
                json_spec_cont = ivy.Container(json_spec)
                parsed_json_cont = ivy.Container(parse_json_to_dict(fpath))
                duplicate_key_chains = list()

                def map_fn(x, kc):
                    if kc in json_spec_cont:
                        duplicate_key_chains.append(kc)
                        return ivy.Container(duplicated={'parent_dir': json_spec_cont[kc], 'this_dir': x})
                    else:
                        return x

                parsed_json_cont = parsed_json_cont.map(map_fn)
                json_spec = {**parsed_json_cont.to_dict(),
                             **json_spec_cont.prune_key_chains(duplicate_key_chains).to_dict()}
            else:
                json_spec = {**parse_json_to_dict(fpath), **json_spec}
        elif os.path.isfile(os.path.join(base_dir, 'reset_to_defaults.sh')):
            pass
        else:
            return json_spec
        base_dir = os.path.abspath(os.path.join(base_dir, '..'))


def get_json_args(json_spec_path, keychains_to_ignore, keychain_to_show, defaults=False, store_duplicates=False):
    if defaults:
        defaults = '.defaults'
    else:
        defaults = ''
    dataset_dirs_args = json_spec_from_fpath(json_spec_path, 'dataset_dirs_args.json' + defaults, store_duplicates)
    dataset_args = json_spec_from_fpath(json_spec_path, 'dataset_args.json' + defaults, store_duplicates)
    data_loader_args = json_spec_from_fpath(json_spec_path, 'data_loader_args.json' + defaults, store_duplicates)
    network_args = json_spec_from_fpath(json_spec_path, 'network_args.json' + defaults, store_duplicates)
    trainer_args = json_spec_from_fpath(json_spec_path, 'trainer_args.json' + defaults, store_duplicates)
    cont = ivy.Container(dataset_dirs_args=dataset_dirs_args,
                         dataset_args=dataset_args,
                         data_loader_args=data_loader_args,
                         network_args=network_args,
                         trainer_args=trainer_args)
    for keychain_to_ignore in keychains_to_ignore:
        if keychain_to_ignore in cont:
            cont[keychain_to_ignore] = 'not_shown'
    if ivy.exists(keychain_to_show):
        cont = cont[keychain_to_show]
    return cont


def print_json_args(base_dir, default_keychains_to_ignore=None):
    ivy.set_framework('numpy')
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--sub_directory', type=str,
                        help='A sub-directory to print the json args for, default is base_dir passed in.')
    parser.add_argument('-dd', '--diff_directory', type=str,
                        help='The directory from which to compare the difference in specifications.')
    parser.add_argument('-kcti', '--keychains_to_ignore', type=str, default=default_keychains_to_ignore,
                        help='A sub-directory to print the json args for, default is the current directory.')
    parser.add_argument('-kcts', '--keychain_to_show', type=str,
                        help='The key-chain to show. Default is None, in which case all key-chains are shown.')
    parser.add_argument('-d', '--show_defaults', action='store_true',
                        help='Whether to show the default json arguments.'
                             'If unset then the current arguments are shown, not the defaut values.')
    parsed_args = parser.parse_args()
    if ivy.exists(parsed_args.sub_directory):
        sub_dir = os.path.abspath(os.path.join(base_dir, parsed_args.sub_directory))
    else:
        sub_dir = base_dir
    if ivy.exists(parsed_args.keychains_to_ignore):
        keychains_to_ignore = [kc[1:-1] for kc in ''.join(parsed_args.keychains_to_ignore[1:-1]).split(',')]
    else:
        keychains_to_ignore = list()
    these_json_args = get_json_args(
        sub_dir, keychains_to_ignore, parsed_args.keychain_to_show, parsed_args.show_defaults, store_duplicates=True)
    if ivy.exists(parsed_args.diff_directory):
        other_sub_dir = os.path.abspath(os.path.join(base_dir, parsed_args.diff_directory))
        if other_sub_dir == sub_dir:
            raise Exception('Invalid diff_directory {} selected, it is the same as the sub_directory {}.'.format(
                other_sub_dir, sub_dir))
        other_json_args = get_json_args(
            other_sub_dir, keychains_to_ignore, parsed_args.keychain_to_show, parsed_args.show_defaults,
            store_duplicates=True)
        diff_keys = 'diff'
        for sub_folder, other_sub_folder in zip(sub_dir.split('/'), other_sub_dir.split('/')):
            if sub_folder != other_sub_folder:
                diff_keys = [sub_folder, other_sub_folder]
                break
        diff_json_args = ivy.Container.diff(these_json_args, other_json_args, diff_keys=diff_keys)
        keyword_color_dict = {'duplicated': 'magenta'}
        if isinstance(diff_keys, list):
            diff_keys_dict = dict(zip(diff_keys, ['red'] * 2))
            keyword_color_dict = {**keyword_color_dict, **diff_keys_dict}
        print(ivy.Container(diff_json_args, keyword_color_dict=keyword_color_dict))
    else:
        print(ivy.Container(these_json_args, keyword_color_dict={'duplicated': 'magenta'}))
    ivy.unset_framework()


def parse_json_to_dict(json_filepath):
    """
    return the data from json file in the form of a python dict
    """
    return_dict = dict()
    with open(json_filepath) as json_data_file:
        loaded_dict = json.load(json_data_file)
    for k, v in loaded_dict.items():
        if k == 'parents':
            rel_fpaths = v
            for rel_fpath in rel_fpaths:
                rel_fpath = os.path.abspath(os.path.join(rel_fpath, json_filepath.split('/')[-1]))
                fpath = os.path.abspath(os.path.join('/'.join(json_filepath.split('/')[:-1]), rel_fpath))
                return_dict = {**return_dict, **parse_json_to_dict(fpath)}
    return {**return_dict, **loaded_dict}


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


def command_line_str_to_spec_dict(spec_str):
    """
    save the python dict as a json file at specified filepath
    """
    if spec_str is not None:
        spec_dict = json.loads(spec_str.replace("'", '"'))
    else:
        spec_dict = {}
    for key in ['dataset_dirs', 'dataset', 'data_loader', 'network', 'trainer', 'tuner']:
        if key not in spec_dict:
            spec_dict[key] = {}
    return spec_dict


def build_dataset_dirs(dataset_dirs_args=None,
                       dataset_dirs_class=None,
                       json_spec_path=None,
                       spec_dict=None):
    """
    build dataset directories specification
    """

    # define dataset directories specification arguments
    if dataset_dirs_args is None:
        dataset_dirs_args = dict()

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'dataset_dirs_args.json')
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['dataset_dirs'] if isinstance(spec_dict, dict) and 'dataset_dirs' in spec_dict else dict()

    # combine args
    dataset_dirs_args = {**json_spec, **this_spec_dict, **dataset_dirs_args}

    # override dataset_dirs_class if specified in dataset_dirs_args
    dataset_dirs_class = ivy.default(ivy.default(
        dataset_dirs_class,
        _import_arg_specified_class_if_present(dataset_dirs_args, 'dataset_dirs_class')),
        DatasetDirs)

    # return dataset directories specification
    return dataset_dirs_class(**dataset_dirs_args)


def build_dataset_spec(dataset_dirs_args=None,
                       dataset_dirs_class=None,
                       dataset_dirs=None,
                       dataset_spec_args=None,
                       dataset_spec_class=None,
                       json_spec_path=None,
                       spec_dict=None):
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
            spec_dict=spec_dict))

    # define dataset specification arguments
    if dataset_spec_args is None:
        dataset_spec_args = dict()
    dataset_spec_args = {**dataset_spec_args, **{'dirs': dataset_dirs}}

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'dataset_args.json')
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['dataset'] if isinstance(spec_dict, dict) and 'dataset' in spec_dict else dict()

    # combine args
    dataset_spec_args = {**json_spec, **this_spec_dict, **dataset_spec_args}

    # override dataset_spec_class if specified in dataset_spec_args
    dataset_spec_class = ivy.default(ivy.default(
        dataset_spec_class,
        _import_arg_specified_class_if_present(dataset_spec_args, 'dataset_spec_class')),
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
                                spec_dict=None):
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
            spec_dict=spec_dict))

    # define network specification arguments
    if network_spec_args is None:
        network_spec_args = dict()
    network_spec_args = {**network_spec_args, **{'dataset_spec': dataset_spec}}

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'network_args.json')
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['network'] if isinstance(spec_dict, dict) and 'network' in spec_dict else dict()

    # combine args
    network_spec_args = {**json_spec, **this_spec_dict, **network_spec_args}

    # override network_spec_class if specified in network_spec_args
    network_spec_class = ivy.default(ivy.default(
        network_spec_class,
        _import_arg_specified_class_if_present(network_spec_args, 'network_spec_class')),
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
                  spec_dict=None):
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
            spec_dict=spec_dict))

    # override network_class if specified in network_spec
    network_class = ivy.default(ivy.default(
        network_class,
        _import_arg_specified_class_if_present(network_spec, 'network_class')),
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
                           spec_dict=None):
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
            spec_dict=spec_dict))

    # define data loader specification arguments
    if data_loader_spec_args is None:
        data_loader_spec_args = dict()
    data_loader_spec_args = {**data_loader_spec_args, **{'dataset_spec': dataset_spec}}

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'data_loader_args.json')
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['data_loader'] if isinstance(spec_dict, dict) and 'data_loader' in spec_dict else dict()

    # combine args
    data_loader_spec_args = {**json_spec, **this_spec_dict, **data_loader_spec_args}

    # override data_loader_spec_class if specified in data_loader_spec_args
    data_loader_spec_class = ivy.default(ivy.default(
        data_loader_spec_class,
        _import_arg_specified_class_if_present(data_loader_spec_args, 'data_loader_spec_class')),
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
                      spec_dict=None):
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
            spec_dict=spec_dict))

    # override data_loader_class if specified in data_loader_spec
    data_loader_class = ivy.default(ivy.default(
        data_loader_class,
        _import_arg_specified_class_if_present(data_loader_spec, 'data_loader_class')),
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
                       spec_dict=None):
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
            spec_dict=spec_dict))

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
            spec_dict=spec_dict))

    # define trainer specification arguments
    if trainer_spec_args is None:
        trainer_spec_args = dict()
    trainer_spec_args = {**trainer_spec_args, **{'data_loader': data_loader,
                                                 'network': network}}

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'trainer_args.json')
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['trainer'] if isinstance(spec_dict, dict) and 'trainer' in spec_dict else dict()

    # combine args
    trainer_spec_args = {**json_spec, **this_spec_dict, **trainer_spec_args}

    # override trainer_spec_class if specified in trainer_spec_args
    trainer_spec_class = ivy.default(ivy.default(
        trainer_spec_class,
        _import_arg_specified_class_if_present(trainer_spec_args, 'trainer_spec_class')),
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
                  spec_dict=None):
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
            spec_dict=spec_dict))

    # override trainer_class if specified in trainer_spec
    trainer_class = ivy.default(ivy.default(
        trainer_class,
        _import_arg_specified_class_if_present(trainer_spec, 'trainer_class')),
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
                     spec_dict=None):
    """
    build tuner specification
    """

    # define dataset directories specification arguments
    if tuner_spec_args is None:
        tuner_spec_args = dict()

    # load json file
    if isinstance(json_spec_path, str):
        json_spec = json_spec_from_fpath(json_spec_path, 'tuner_args.json')
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['tuner'] if isinstance(spec_dict, dict) and 'tuner' in spec_dict else dict()

    # combine args
    tuner_spec_args = {**json_spec, **this_spec_dict, **tuner_spec_args}

    # override tuner_spec_class if specified in tuner_spec_args
    tuner_spec_class = ivy.default(ivy.default(
        tuner_spec_class,
        _import_arg_specified_class_if_present(tuner_spec_args, 'tuner_spec_class')),
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
            spec_dict=spec_dict))

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
                spec_dict=None):
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
            spec_dict=spec_dict))

    # override tuner_class if specified in tuner_spec_args
    tuner_class = ivy.default(ivy.default(
        tuner_class, _import_arg_specified_class_if_present(tuner_spec, 'tuner_class')),
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
        spec_dict=spec_dict)
