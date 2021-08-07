# global
import os
import ivy
import json
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


def parse_json_to_dict(json_filepath):
    """
    return the data from json file in the form of a python dict
    """
    return_dict = dict()
    with open(json_filepath) as json_data_file:
        loaded_dict = json.load(json_data_file)
    for k, v in loaded_dict.items():
        if k == 'parent':
            rel_fpath = v
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
                       dataset_dirs_class=DatasetDirs,
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
        fpath = os.path.join(json_spec_path, 'dataset_dirs_args.json')
        json_spec = parse_json_to_dict(fpath)
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['dataset_dirs'] if isinstance(spec_dict, dict) and 'dataset_dirs' in spec_dict else dict()

    # combine args
    dataset_dirs_args = {**json_spec, **this_spec_dict, **dataset_dirs_args}

    # override dataset_dirs_class if specified in dataset_dirs_args
    dataset_dirs_class = ivy.default(
        _import_arg_specified_class_if_present(dataset_dirs_args, 'dataset_dirs_class'), dataset_dirs_class)

    # return dataset directories specification
    # noinspection PyArgumentList
    return dataset_dirs_class(**dataset_dirs_args)


def build_dataset_spec(dataset_dirs_args=None,
                       dataset_dirs_class=DatasetDirs,
                       dataset_spec_args=None,
                       dataset_spec_class=DatasetSpec,
                       json_spec_path=None,
                       spec_dict=None):
    """
    build dataset specification
    """

    # build dataset directories
    dataset_dirs = build_dataset_dirs(dataset_dirs_args,
                                      dataset_dirs_class,
                                      json_spec_path,
                                      spec_dict)

    # define dataset specification arguments
    if dataset_spec_args is None:
        dataset_spec_args = dict()
    dataset_spec_args = {**dataset_spec_args, **{'dirs': dataset_dirs}}

    # load json file
    if isinstance(json_spec_path, str):
        fpath = os.path.join(json_spec_path, 'dataset_args.json')
        json_spec = parse_json_to_dict(fpath)
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['dataset'] if isinstance(spec_dict, dict) and 'dataset' in spec_dict else dict()

    # combine args
    dataset_spec_args = {**json_spec, **this_spec_dict, **dataset_spec_args}

    # override dataset_spec_class if specified in dataset_spec_args
    dataset_spec_class = ivy.default(
        _import_arg_specified_class_if_present(dataset_spec_args, 'dataset_spec_class'), dataset_spec_class)

    # return dataset specification
    return dataset_spec_class(**dataset_spec_args)


def build_network_specification(network_spec_args=None,
                                network_spec_class=NetworkSpec,
                                json_spec_path=None,
                                spec_dict=None):
    """
    build network specification
    """

    # define network specification arguments
    if network_spec_args is None:
        network_spec_args = dict()
    network_spec_args = {**network_spec_args}

    # load json file
    if isinstance(json_spec_path, str):
        fpath = os.path.join(json_spec_path, 'network_args.json')
        json_spec = parse_json_to_dict(fpath)
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['network'] if isinstance(spec_dict, dict) and 'network' in spec_dict else dict()

    # combine args
    network_spec_args = {**json_spec, **this_spec_dict, **network_spec_args}

    # override network_spec_class if specified in network_spec_args
    network_spec_class = ivy.default(
        _import_arg_specified_class_if_present(network_spec_args, 'network_spec_class'), network_spec_class)

    # return network
    # noinspection PyArgumentList
    return network_spec_class(**network_spec_args)


def build_network(network_class=None,
                  network_spec_args=None,
                  network_spec_class=NetworkSpec,
                  json_spec_path=None,
                  spec_dict=None):
    """
    build network
    """

    # build network specification
    network_spec = build_network_specification(network_spec_args,
                                               network_spec_class,
                                               json_spec_path,
                                               spec_dict)

    # override network_class if specified in network_spec
    network_class = ivy.default(
        _import_arg_specified_class_if_present(network_spec, 'network_class'), network_class)

    # verify network_class exists
    if not ivy.exists(network_class):
        raise Exception('network_class must either be specified in this build_network() method,'
                        'or network_class attribute must be specified in the network_spec instance')

    # network
    return network_class(network_spec)


def build_data_loader_spec(network_class=None,
                           dataset_dirs_args=None,
                           dataset_dirs_class=DatasetDirs,
                           dataset_spec_args=None,
                           dataset_spec_class=DatasetSpec,
                           data_loader_spec_args=None,
                           data_loader_spec_class=DataLoaderSpec,
                           network_spec_args=None,
                           network_spec_class=NetworkSpec,
                           json_spec_path=None,
                           spec_dict=None):
    """
    build data loader specification
    """

    # build dataset specification
    dataset_spec = build_dataset_spec(dataset_dirs_args,
                                      dataset_dirs_class,
                                      dataset_spec_args,
                                      dataset_spec_class,
                                      json_spec_path,
                                      spec_dict)

    # build network
    network = build_network(network_class,
                            network_spec_args,
                            network_spec_class,
                            json_spec_path,
                            spec_dict)

    # define data loader specification arguments
    if data_loader_spec_args is None:
        data_loader_spec_args = dict()
    data_loader_spec_args = {**data_loader_spec_args, **{'dataset_spec': dataset_spec, 'network': network}}

    # load json file
    if isinstance(json_spec_path, str):
        fpath = os.path.join(json_spec_path, 'data_loader_args.json')
        json_spec = parse_json_to_dict(fpath)
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['data_loader'] if isinstance(spec_dict, dict) and 'data_loader' in spec_dict else dict()

    # combine args
    data_loader_spec_args = {**json_spec, **this_spec_dict, **data_loader_spec_args}

    # override data_loader_spec_class if specified in data_loader_spec_args
    data_loader_spec_class = ivy.default(
        _import_arg_specified_class_if_present(data_loader_spec_args, 'data_loader_spec_class'), data_loader_spec_class)

    # return data loader
    return data_loader_spec_class(**data_loader_spec_args)


def build_data_loader(data_loader_class=None,
                      network_class=None,
                      dataset_dirs_args=None,
                      dataset_dirs_class=DatasetDirs,
                      dataset_spec_args=None,
                      dataset_spec_class=DatasetSpec,
                      data_loader_spec_args=None,
                      data_loader_spec_class=DataLoaderSpec,
                      network_spec_args=None,
                      network_spec_class=NetworkSpec,
                      json_spec_path=None,
                      spec_dict=None):
    """
    build data loader
    """

    # build data loader specification
    data_loader_spec = build_data_loader_spec(network_class,
                                              dataset_dirs_args,
                                              dataset_dirs_class,
                                              dataset_spec_args,
                                              dataset_spec_class,
                                              data_loader_spec_args,
                                              data_loader_spec_class,
                                              network_spec_args,
                                              network_spec_class,
                                              json_spec_path,
                                              spec_dict)

    # override data_loader_class if specified in data_loader_spec
    data_loader_class = ivy.default(
        _import_arg_specified_class_if_present(data_loader_spec, 'data_loader_class'), data_loader_class)

    # verify data_loader_class exists
    if not ivy.exists(data_loader_class):
        raise Exception('data_loader_class must either be specified in this build_data_loader() method,'
                        'or data_loader_class attribute must be specified in the data_loader_spec instance')

    # return data loader
    return data_loader_class(data_loader_spec)


def build_trainer_spec(data_loader_class=None,
                       network_class=None,
                       dataset_dirs_args=None,
                       dataset_dirs_class=DatasetDirs,
                       dataset_spec_args=None,
                       dataset_spec_class=DatasetSpec,
                       data_loader_spec_args=None,
                       data_loader_spec_class=DataLoaderSpec,
                       network_spec_args=None,
                       network_spec_class=NetworkSpec,
                       trainer_spec_args=None,
                       trainer_spec_class=TrainerSpec,
                       json_spec_path=None,
                       spec_dict=None):
    """
    build trainer specification
    """

    # build data loader
    data_loader = build_data_loader(data_loader_class,
                                    network_class,
                                    dataset_dirs_args,
                                    dataset_dirs_class,
                                    dataset_spec_args,
                                    dataset_spec_class,
                                    data_loader_spec_args,
                                    data_loader_spec_class,
                                    network_spec_args,
                                    network_spec_class,
                                    json_spec_path,
                                    spec_dict)

    # define trainer specification arguments
    if trainer_spec_args is None:
        trainer_spec_args = dict()
    trainer_spec_args = {**trainer_spec_args, **{'data_loader': data_loader,
                                                 'network': data_loader.spec.network}}

    # load json file
    if isinstance(json_spec_path, str):
        fpath = os.path.join(json_spec_path, 'trainer_args.json')
        json_spec = parse_json_to_dict(fpath)
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['trainer'] if isinstance(spec_dict, dict) and 'trainer' in spec_dict else dict()

    # combine args
    trainer_spec_args = {**json_spec, **this_spec_dict, **trainer_spec_args}

    # override trainer_spec_class if specified in trainer_spec_args
    trainer_spec_class = ivy.default(
        _import_arg_specified_class_if_present(trainer_spec_args, 'trainer_spec_class'), trainer_spec_class)

    # return trainer specification
    return trainer_spec_class(**trainer_spec_args)


def build_trainer(data_loader_class=None,
                  network_class=None,
                  trainer_class=None,
                  dataset_dirs_args=None,
                  dataset_dirs_class=DatasetDirs,
                  dataset_spec_args=None,
                  dataset_spec_class=DatasetSpec,
                  data_loader_spec_args=None,
                  data_loader_spec_class=DataLoaderSpec,
                  network_spec_args=None,
                  network_spec_class=NetworkSpec,
                  trainer_spec_args=None,
                  trainer_spec_class=TrainerSpec,
                  json_spec_path=None,
                  spec_dict=None):
    """
    build trainer
    """

    # build trainer spec
    trainer_spec = build_trainer_spec(data_loader_class,
                                      network_class,
                                      dataset_dirs_args,
                                      dataset_dirs_class,
                                      dataset_spec_args,
                                      dataset_spec_class,
                                      data_loader_spec_args,
                                      data_loader_spec_class,
                                      network_spec_args,
                                      network_spec_class,
                                      trainer_spec_args,
                                      trainer_spec_class,
                                      json_spec_path,
                                      spec_dict)

    # override trainer_class if specified in trainer_spec
    trainer_class = ivy.default(
        _import_arg_specified_class_if_present(trainer_spec, 'trainer_class'), trainer_class)

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
                     dataset_dirs_class=DatasetDirs,
                     dataset_spec_args=None,
                     dataset_spec_class=DatasetSpec,
                     data_loader_spec_args=None,
                     data_loader_spec_class=DataLoaderSpec,
                     network_spec_args=None,
                     network_spec_class=NetworkSpec,
                     trainer_spec_args=None,
                     trainer_spec_class=TrainerSpec,
                     tuner_spec_args=None,
                     tuner_spec_class=TunerSpec,
                     json_spec_path=None,
                     spec_dict=None):
    """
    build tuner specification
    """
    trainer = build_trainer(data_loader_class,
                            network_class,
                            trainer_class,
                            dataset_dirs_args,
                            dataset_dirs_class,
                            dataset_spec_args,
                            dataset_spec_class,
                            data_loader_spec_args,
                            data_loader_spec_class,
                            network_spec_args,
                            network_spec_class,
                            trainer_spec_args,
                            trainer_spec_class,
                            json_spec_path,
                            spec_dict)

    # define dataset directories specification arguments
    if tuner_spec_args is None:
        tuner_spec_args = dict()

    # load json file
    if isinstance(json_spec_path, str):
        fpath = os.path.join(json_spec_path, 'tuner_args.json')
        json_spec = parse_json_to_dict(fpath)
    else:
        json_spec = dict()

    # load from spec dict
    this_spec_dict =\
        spec_dict['tuner'] if isinstance(spec_dict, dict) and 'tuner' in spec_dict else dict()

    # combine args
    tuner_spec_args = {**json_spec, **this_spec_dict, **tuner_spec_args}

    # override tuner_spec_class if specified in tuner_spec_args
    tuner_spec_class = ivy.default(
        _import_arg_specified_class_if_present(tuner_spec_args, 'tuner_spec_class'), tuner_spec_class)

    # return tuner specification
    return tuner_spec_class(trainer,
                            **tuner_spec_args)


def build_tuner(data_loader_class=None,
                network_class=None,
                trainer_class=None,
                dataset_dirs_args=None,
                dataset_dirs_class=DatasetDirs,
                dataset_spec_args=None,
                dataset_spec_class=DatasetSpec,
                data_loader_spec_args=None,
                data_loader_spec_class=DataLoaderSpec,
                network_spec_args=None,
                network_spec_class=NetworkSpec,
                trainer_spec_args=None,
                trainer_spec_class=TrainerSpec,
                tuner_spec_args=None,
                tuner_spec_class=TunerSpec,
                tuner_class=Tuner,
                json_spec_path=None,
                spec_dict=None):
    """
    build tuner
    """

    # override tuner_class if specified in tuner_spec_args
    tuner_class = ivy.default(
        _import_arg_specified_class_if_present(tuner_spec_args, 'tuner_class'), tuner_class)

    # return tuner
    return tuner_class(data_loader_class,
                       network_class,
                       trainer_class,
                       dataset_dirs_args,
                       dataset_dirs_class,
                       dataset_spec_args,
                       dataset_spec_class,
                       data_loader_spec_args,
                       data_loader_spec_class,
                       network_spec_args,
                       network_spec_class,
                       trainer_spec_args,
                       trainer_spec_class,
                       tuner_spec_args,
                       tuner_spec_class,
                       json_spec_path,
                       spec_dict)
