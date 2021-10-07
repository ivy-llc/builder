# global
import os
import ivy
import math
import logging
import numpy as np
import multiprocessing
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
except ModuleNotFoundError:
    ray = None
    tune = None

# local
from ivy.core.container import Container
from ivy_builder import builder as builder_module
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs.data_loader_spec import DataLoaderSpec
from ivy_builder.specs.network_spec import NetworkSpec
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.specs.tuner_spec import TunerSpec

SPEC_KEYS = ['dataset_dirs', 'dataset_spec', 'data_loader_spec', 'network_spec', 'trainer_spec']
SHORT_SPEC_KEYS_DICT = {'dataset_dirs': 'dd',
                        'dataset_spec': 'ds',
                        'data_loader_spec': 'ls',
                        'network_spec': 'ns',
                        'trainer_spec': 'ts'}
FIXED_CONFIG_KEYS = ['train_steps_per_tune_step', 'framework']


def _is_numeric_leaf(val):
    if not isinstance(val, Container):
        return False
    if val.if_exists('min') and val.if_exists('max'):
        return True
    return False


def _is_config_leaf(val):
    if not isinstance(val, Container):
        return False
    if val.if_exists('configs'):
        return True
    return False


def _is_leaf(val):
    return _is_numeric_leaf(val) or _is_config_leaf(val)


def _convert_numeric_leaf(val):
    min_val = val.min
    max_val = val.max
    mean_val = (max_val + min_val) / 2
    sd_val = max_val - mean_val
    gaussian = val.if_exists('gaussian')
    grid = val.if_exists('grid')
    uniform = val.if_exists('uniform')
    selections = [bool(gaussian), bool(grid), bool(uniform)]
    num_selected = sum(selections)
    if num_selected > 1:
        raise Exception('only one of [ gaussian | grid | uniform ] can be selected,'
                        'but {} are selected, with the following options set: {}'.format(num_selected, selections))
    if num_selected == 0:
        uniform = True
    if val.if_exists('exponent'):
        exponential = True
        exponent = val.exponent
        log_exponent = np.log(exponent)
    else:
        exponential = False
    as_int = val.if_exists('as_int')
    if gaussian:
        if exponential:
            if as_int:
                sample_func = lambda spec, m=mean_val, s=sd_val: \
                    int(np.round(exponent ** (np.random.normal(np.log(m) / log_exponent,
                                                               np.log(s) / log_exponent))))
            else:
                sample_func = lambda spec, m=mean_val, s=sd_val: \
                    exponent ** (np.random.normal(np.log(m) / log_exponent,
                                                  np.log(s) / log_exponent))
        else:
            if as_int:
                sample_func = lambda spec, m=mean_val, s=sd_val: \
                    int(np.round(np.random.normal(m, s)))
            else:
                sample_func = lambda spec, m=mean_val, s=sd_val: \
                    np.random.normal(m, s)
        numeric_leaf = tune.sample_from(sample_func)
    elif uniform:
        if exponential:
            if as_int:
                sample_func = lambda spec, mi=min_val, ma=max_val: \
                    int(np.round(exponent ** (np.random.uniform(np.log(mi) / log_exponent,
                                                                np.log(ma) / log_exponent))))
            else:
                sample_func = lambda spec, mi=min_val, ma=max_val: \
                    exponent ** (np.random.uniform(np.log(mi) / log_exponent,
                                                   np.log(ma) / log_exponent))
        else:
            if as_int:
                sample_func = lambda spec, mi=min_val, ma=max_val: int(np.round(np.random.uniform(mi, ma)))
            else:
                sample_func = lambda spec, mi=min_val, ma=max_val: np.random.uniform(mi, ma)
        numeric_leaf = tune.sample_from(sample_func)
    else:  # grid
        num_grid_samples = val.num_grid_samples
        if exponential:
            if as_int:
                # noinspection PyUnboundLocalVariable
                grid_vals = np.round(exponent ** (np.linspace(
                    np.log(min_val) / log_exponent,
                    np.log(max_val) / log_exponent,
                    num_grid_samples))).astype(np.int32).tolist()
            else:
                # noinspection PyUnboundLocalVariable
                grid_vals = (exponent ** (np.linspace(np.log(min_val) / log_exponent,
                                                      np.log(max_val) / log_exponent,
                                                      num_grid_samples))).tolist()
        else:
            if as_int:
                grid_vals = np.round(np.linspace(min_val, max_val, num_grid_samples)).astype(np.uint32).tolist()
            else:
                grid_vals = np.linspace(min_val, max_val, num_grid_samples).tolist()
        numeric_leaf = tune.grid_search(grid_vals)
    return numeric_leaf


def _convert_config_leaf(val):
    if val.if_exists('grid'):
        return tune.grid_search(val.configs)
    return tune.sample_from(lambda: np.random.choice(val.configs))


def _convert_multi_config_leaf(keys, val):
    new_config = Container()
    new_config.grid = val.if_exists('grid')
    new_config.configs = [dict(zip(keys, v)) for v in val.configs]
    return _convert_config_leaf(new_config)


# noinspection PyUnboundLocalVariable
def _convert_tuner_spec(spec, key_chain=''):
    new_spec = Container()
    for i, (key, val) in enumerate(spec.items()):
        key_chain = (key_chain + '/' + key) if key_chain != '' else key
        spec_key = [sk for sk in SPEC_KEYS if sk in key_chain]
        if not spec_key:
            new_spec[key] = val
            continue
        if not _is_leaf(val):
            if not isinstance(val, Container):
                new_spec[key] = val
            else:
                new_spec[key] = _convert_tuner_spec(val, key_chain)
            continue
        if _is_numeric_leaf(val):
            new_spec[key] = _convert_numeric_leaf(val)
        elif _is_config_leaf(val):
            keys = key.split('_AND_')
            if len(keys) == 1:
                new_spec[keys[0]] = _convert_config_leaf(val)
            else:
                new_spec[key] = _convert_multi_config_leaf(keys, val)
        else:
            raise Exception('invalid leaf')
    return new_spec


class Tuner:

    def __init__(self,
                 data_loader_class,
                 network_class,
                 trainer_class,
                 dataset_dirs_args: dict = None,
                 dataset_dirs_class: DatasetDirs.__base__ = DatasetDirs,
                 dataset_dirs: DatasetDirs = None,
                 dataset_spec_args: dict = None,
                 dataset_spec_class: DatasetSpec.__base__ = DatasetSpec,
                 dataset_spec: DatasetSpec = None,
                 data_loader_spec_args: dict = None,
                 data_loader_spec_class: DataLoaderSpec.__base__ = DataLoaderSpec,
                 data_loader_spec: DataLoaderSpec = None,
                 data_loader=None,
                 network_spec_args: dict = None,
                 network_spec_class: NetworkSpec.__base__ = NetworkSpec,
                 network_spec: NetworkSpec = None,
                 network=None,
                 trainer_spec_args: dict = None,
                 trainer_spec_class: TrainerSpec.__base__ = TrainerSpec,
                 trainer_spec: TrainerSpec = None,
                 trainer=None,
                 tuner_spec_args: dict = None,
                 tuner_spec_class: TunerSpec.__base__ = TunerSpec,
                 tuner_spec: TunerSpec = None,
                 json_spec_path: str = None,
                 spec_cont: dict = None):
        """
        base class for any tune trainers
        """
        if not ivy.exists(tune):
            raise Exception('ray[tune] is needed in order to use the Tuner class, but it is not installed.'
                            'Please install via pip install ray[tune]')
        self._data_loader_class = data_loader_class
        self._network_class = network_class
        self._trainer_class = trainer_class
        self._dataset_dirs_args = ivy.default(dataset_dirs_args, dict())
        self._dataset_dirs_class = dataset_dirs_class
        self._dataset_dirs = dataset_dirs
        self._dataset_spec_args = ivy.default(dataset_spec_args, dict())
        self._dataset_spec_class = dataset_spec_class
        self._dataset_spec = dataset_spec
        self._data_loader_spec_args = ivy.default(data_loader_spec_args, dict())
        self._data_loader_spec_class = data_loader_spec_class
        self._data_loader_spec = data_loader_spec
        self._data_loader = data_loader
        self._network_spec_args = ivy.default(network_spec_args, dict())
        self._network_spec_class = network_spec_class
        self._network_spec = network_spec
        self._network = network
        self._trainer_spec_args = ivy.default(trainer_spec_args, dict())
        self._trainer_spec_class = trainer_spec_class
        self._trainer_spec = trainer_spec
        self._trainer = trainer
        self._tuner_spec_args = ivy.default(tuner_spec_args, dict())
        self._tuner_spec_class = tuner_spec_class
        self._tuner_spec = tuner_spec
        self._json_spec_path = json_spec_path
        self._spec_cont = spec_cont

        # initialized on _setup
        self._trainer = None

        # builder
        while len(ivy.framework_stack) > 0:
            logging.info('unsetting framework {}, framework stack must be empty when'
                         'initializing tuner class.'.format(ivy.framework_stack[-1]))
            ivy.unset_framework()
        self._builder = builder_module

        # tuner spec
        self._spec = self._builder.build_tuner_spec(
            data_loader_class=self._data_loader_class,
            network_class=self._network_class,
            trainer_class=self._trainer_class,
            dataset_dirs_args=self._dataset_dirs_args,
            dataset_dirs_class=self._dataset_dirs_class,
            dataset_dirs=self._dataset_dirs,
            dataset_spec_args=self._dataset_spec_args,
            dataset_spec_class=self._dataset_spec_class,
            dataset_spec=self._dataset_spec,
            data_loader_spec_args=self._data_loader_spec_args,
            data_loader_spec_class=self._data_loader_spec_class,
            data_loader_spec=self._data_loader_spec,
            data_loader=self._data_loader,
            network_spec_args=self._network_spec_args,
            network_spec_class=self._network_spec_class,
            network_spec=self._network_spec,
            network=self._network,
            trainer_spec_args=self._trainer_spec_args,
            trainer_spec_class=self._trainer_spec_class,
            trainer_spec=self._trainer_spec,
            trainer=self._trainer,
            tuner_spec_args=self._tuner_spec_args,
            tuner_spec_class=self._tuner_spec_class,
            json_spec_path=self._json_spec_path,
            spec_cont=self._spec_cont)
        self._spec = _convert_tuner_spec(self._spec)

    def tune(self):

        # Create Trainable class #
        # -----------------------#

        # builder for TuneTrainable
        builder = self._builder

        # classes and args for TuneTrainable
        data_loader_class = self._data_loader_class
        network_class = self._network_class
        trainer_class = self._trainer_class
        dataset_dirs_args = self._dataset_dirs_args
        dataset_dirs_class = self._dataset_dirs_class
        dataset_dirs = self._dataset_dirs
        dataset_spec_args = self._dataset_spec_args
        dataset_spec_class = self._dataset_spec_class
        dataset_spec = self._dataset_spec
        data_loader_spec_args = self._data_loader_spec_args
        data_loader_spec_class = self._data_loader_spec_class
        data_loader_spec = self._data_loader_spec
        data_loader = self._data_loader
        network_spec_args = self._network_spec_args
        network_spec_class = self._network_spec_class
        network_spec = self._network_spec
        network = self._network
        trainer_spec_args = self._trainer_spec_args
        trainer_spec_class = self._trainer_spec_class
        trainer_spec = self._trainer_spec
        json_spec_path = self._json_spec_path
        spec_cont = self._spec_cont
        orig_log_dir = self._spec.trainer.spec.log_dir

        # noinspection PyAttributeOutsideInit
        class TuneTrainable(tune.Trainable):

            def setup(self, _):
                ivy.set_framework(self.config['framework'])
                self._train_steps_per_tune_step = self.config['train_steps_per_tune_step']
                config_cont = Container(self.config)
                self._config_str = '_'.join(
                    [str(SHORT_SPEC_KEYS_DICT[kc.split('/')[0]]) + '_' + kc.split('/')[-1] + '_' +
                     ("%.2g" % val if isinstance(val, float) else str(val)) for kc, val in config_cont.to_iterator()
                     if (isinstance(val, (float, int, bool, type(None))) and kc not in FIXED_CONFIG_KEYS)])
                trainer_spec_args['log_dir'] = os.path.join(orig_log_dir, self._config_str)
                new_args = dict()
                for class_key, args in zip(SPEC_KEYS, [dataset_dirs_args, dataset_spec_args, data_loader_spec_args,
                                                       network_spec_args, trainer_spec_args]):
                    new_args[class_key] =\
                        Container({**args, **(self.config[class_key] if
                                              class_key in self.config else {})}).prune_key_from_key_chains(
                            containing='_AND_')

                self._trainer = builder.build_trainer(data_loader_class=data_loader_class,
                                                      network_class=network_class,
                                                      trainer_class=trainer_class,
                                                      dataset_dirs_args=new_args['dataset_dirs'],
                                                      dataset_dirs_class=dataset_dirs_class,
                                                      dataset_dirs=dataset_dirs,
                                                      dataset_spec_args=new_args['dataset_spec'],
                                                      dataset_spec_class=dataset_spec_class,
                                                      dataset_spec=dataset_spec,
                                                      data_loader_spec_args=new_args['data_loader_spec'],
                                                      data_loader_spec_class=data_loader_spec_class,
                                                      data_loader_spec=data_loader_spec,
                                                      data_loader=data_loader,
                                                      network_spec_args=new_args['network_spec'],
                                                      network_spec_class=network_spec_class,
                                                      network_spec=network_spec,
                                                      network=network,
                                                      trainer_spec_args=new_args['trainer_spec'],
                                                      trainer_spec_class=trainer_spec_class,
                                                      trainer_spec=trainer_spec,
                                                      json_spec_path=json_spec_path,
                                                      spec_cont=spec_cont)
                # unset at_end configs
                self._save_at_end = self._trainer.spec.save_at_end
                self._trainer.spec.save_at_end = False
                self._log_at_end = self._trainer.spec.log_at_end
                self._trainer.spec.log_at_end = False
                self._vis_at_end = self._trainer.spec.vis_at_end
                self._trainer.spec.vis_at_end = False

                self._trainer.setup()
                # noinspection PyProtectedMember
                self._trainer_global_step = self._trainer._starting_iteration
                self._trainer_total_iterations = self._trainer.spec.total_iterations
                self.timestep = int(math.floor(self._trainer_global_step / self._train_steps_per_tune_step))

            # noinspection PyProtectedMember
            def step(self):
                total_iterations = min(self._trainer_global_step + self._train_steps_per_tune_step,
                                       self._trainer_total_iterations)
                self._trainer_global_step = self._trainer.train(self._trainer_global_step, total_iterations)
                self.timestep += 1
                ret_dict = {'timestep': self.timestep,
                            'cost': ivy.to_numpy(self._trainer.moving_average_loss)}
                if self._trainer_global_step >= self._trainer_total_iterations:
                    if self._save_at_end:
                        self._trainer._save()
                    if self._log_at_end and ivy.exists(self._trainer._training_batch):
                        self._trainer._log_scalars()
                    if self._vis_at_end:
                        dl = self._trainer.spec.data_loader
                        net = self._trainer.spec.network
                        tb = self._trainer._training_batch
                        gs = self._trainer._global_step
                        self._trainer._write_image_summaries(dl, net, tb, gs)
                    ret_dict[tune.result.DONE] = True
                return ret_dict

            def save_checkpoint(self, checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_name = 'step_{}'.format(self.timestep)
                save_path = os.path.join(checkpoint_dir, save_name)
                self._trainer.save(save_path)
                print('saved checkpoint to path: {}'.format(save_path))
                return save_path

            def load_checkpoint(self, checkpoint_path):
                self._trainer.restore(checkpoint_path, self._trainer_global_step)
                print('loaded checkpoint from {}'.format(checkpoint_path))

            def cleanup(self):
                self._trainer.close()
                ivy.unset_framework()

        # Run this trainable class #
        # -------------------------#

        max_t = int(np.ceil(self._spec.trainer.spec.total_iterations/self._spec.train_steps_per_tune_step))
        ahb = AsyncHyperBandScheduler(
            time_attr="timestep",
            metric="cost",
            mode="min",
            grace_period=max_t if self._spec.grace_period == -1 else self._spec.grace_period,
            max_t=max_t)

        num_cpus = multiprocessing.cpu_count()
        assert num_cpus > 0
        num_gpus = ivy.num_gpus()
        cpus_per_trial = num_cpus/self._spec.parallel_trials
        gpus_per_trial = num_gpus/self._spec.parallel_trials
        if self._spec.device_priority == 'cpu' or num_gpus == 0:
            cpus_per_trial = int(round(cpus_per_trial)) if cpus_per_trial > 1 else cpus_per_trial
            parallel_trials = math.floor(num_cpus / cpus_per_trial)
            gpus_per_trial = num_gpus / parallel_trials
            gpus_per_trial = math.floor(gpus_per_trial) if gpus_per_trial > 1 else gpus_per_trial
        elif self._spec.device_priority == 'gpu':
            gpus_per_trial = int(round(gpus_per_trial)) if gpus_per_trial > 1 else gpus_per_trial
            parallel_trials = math.floor(num_gpus / gpus_per_trial)
            cpus_per_trial = num_cpus / parallel_trials
            cpus_per_trial = math.floor(cpus_per_trial) if cpus_per_trial > 1 else cpus_per_trial
        else:
            raise Exception('device_priority must be one of [ cpu | gpu ], but found {}'.format(
                self._spec.device_priority))
        ivy.unset_framework()

        reporter = CLIReporter(['cost'])

        # initialize ray with custom temp_dir
        ray.init(_temp_dir=os.path.join('/'.join(self._spec.trainer.spec.log_dir.split('/')[:-1]), 'ray'),
                 ignore_reinit_error=True)

        return tune.run(
            TuneTrainable,
            progress_reporter=reporter,
            name=self._spec.name,
            scheduler=ahb,
            stop={"timestep":
                      int(np.ceil(self._spec.trainer.spec.total_iterations/self._spec.train_steps_per_tune_step))},
            num_samples=self._spec.num_samples,
            resources_per_trial={
                "cpu": cpus_per_trial,
                "gpu": gpus_per_trial
            },
            config={key: val for key, val in self._spec.items()
                    if (isinstance(val, dict) or isinstance(val, tune.sample.Function)
                        or key in ['framework', 'train_steps_per_tune_step'])},
            local_dir='/'.join(self._spec.trainer.spec.log_dir.split('/')[:-1]),
            checkpoint_freq=self._spec.checkpoint_freq,
            checkpoint_at_end=True)

    def close(self) -> None:
        """
        Close this tuner, and destroy all child objects or processes which may not be garbage collected.
        """
        pass
