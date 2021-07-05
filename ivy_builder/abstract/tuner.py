# global
import os
import ivy
import json
import numpy as np
from ray import tune
import multiprocessing
from ray.tune import CLIReporter
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler

# local
from ivy.core.container import Container
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs.data_loader_spec import DataLoaderSpec
from ivy_builder.specs.network_spec import NetworkSpec
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.specs.tuner_spec import TunerSpec

SPEC_KEYS = ['dd', 'ds', 'dls', 'ns', 'ts']


def _convert_tune_config(config):
    return_dict = dict()

    for i, (key, arg) in enumerate(config.items()):
        spec_key = [spec_key for spec_key in SPEC_KEYS if spec_key in key]
        if not spec_key:
            raise Exception('Invalid spec key found in config.')
        min_val = arg['min']
        max_val = arg['max']
        mean_val = (max_val + min_val) / 2
        sd_val = max_val - mean_val
        exponential = arg['exponential']
        with_gaussian = arg['gaussian']
        as_int = arg['as_int']
        if with_gaussian:
            if exponential:
                if as_int:
                    sample_func = lambda spec, m=mean_val, s=sd_val: int(np.round(10 ** np.random.normal(m, s)))
                else:
                    sample_func = lambda spec, m=mean_val, s=sd_val: 10 ** np.random.normal(m, s)
            else:
                if as_int:
                    sample_func = lambda spec, m=mean_val, s=sd_val: int(np.round(np.random.normal(m, s)))
                else:
                    sample_func = lambda spec, m=mean_val, s=sd_val: np.random.normal(m, s)
        else:
            if exponential:
                if as_int:
                    sample_func = \
                        lambda spec, mi=min_val, ma=max_val: int(np.round(10 ** np.random.uniform(mi, ma)))
                else:
                    sample_func = lambda spec, mi=min_val, ma=max_val: 10 ** np.random.uniform(mi, ma)
            else:
                if as_int:
                    sample_func = \
                        lambda spec, mi=min_val, ma=max_val: int(np.round(np.random.uniform(mi, ma)))
                else:
                    sample_func = lambda spec, mi=min_val, ma=max_val: np.random.uniform(mi, ma)
        return_dict[key] = tune.sample_from(sample_func)

    return Container(return_dict)


class Tuner:

    def __init__(self,
                 data_loader_class,
                 network_class,
                 trainer_class,
                 dataset_dirs_args: dict = None,
                 dataset_dirs_class: DatasetDirs.__base__ = DatasetDirs,
                 dataset_spec_args: dict = None,
                 dataset_spec_class: DatasetSpec.__base__ = DatasetSpec,
                 data_loader_spec_args: dict = None,
                 data_loader_spec_class: DataLoaderSpec.__base__ = DataLoaderSpec,
                 network_spec_args: dict = None,
                 network_spec_class: NetworkSpec.__base__ = NetworkSpec,
                 trainer_spec_args: dict = None,
                 trainer_spec_class: TrainerSpec.__base__ = TrainerSpec,
                 tuner_spec_args: dict = None,
                 tuner_spec_class: TrainerSpec.__base__ = TunerSpec):
        """
        base class for any tune trainers
        """
        self._data_loader_class = data_loader_class
        self._network_class = network_class
        self._trainer_class = trainer_class
        self._dataset_dirs_args = dataset_dirs_args
        self._dataset_dirs_class = dataset_dirs_class
        self._dataset_spec_args = dataset_spec_args
        self._dataset_spec_class = dataset_spec_class
        self._data_loader_spec_args = data_loader_spec_args
        self._data_loader_spec_class = data_loader_spec_class
        self._network_spec_args = network_spec_args
        self._network_spec_class = network_spec_class
        self._trainer_spec_args = trainer_spec_args
        self._trainer_spec_class = trainer_spec_class

        # initialized on _setup
        self._trainer = None

        # builder
        import ivy_builder.builder as builder
        self._builder = builder

    def tune(self, name, config, num_samples, max_t, num_gpus):

        data_loader_class = self._data_loader_class
        network_class = self._network_class
        trainer_class = self._trainer_class
        dataset_dirs_args = self._dataset_dirs_args
        dataset_dirs_class = self._dataset_dirs_class
        dataset_spec_args = self._dataset_spec_args
        dataset_spec_class = self._dataset_spec_class
        data_loader_spec_args = self._data_loader_spec_args
        data_loader_spec_class = self._data_loader_spec_class
        network_spec_args = self._network_spec_args
        network_spec_class = self._network_spec_class
        trainer_spec_args = self._trainer_spec_args
        trainer_spec_class = self._trainer_spec_class

        builder = self._builder

        # Create Trainable class #
        # -----------------------#

        class TuneTrainable(tune.Trainable):

            def setup(self, _):
                self.timestep = 0
                self._trainer_global_step = 0
                ivy.set_framework(self.config['framework'])
                self._train_steps_per_tune_step = self.config['train_steps_per_tune_step']

                new_args = dict()
                for class_key, args in zip(SPEC_KEYS, [dataset_dirs_args, dataset_spec_args, data_loader_spec_args,
                                                       network_spec_args, trainer_spec_args]):
                    if args:
                        new_args[class_key] = dict([(key, self.config[class_key + '_' + key])
                                                    if class_key + '_' + key in self.config
                                                    else (key, value) for key, value in args.items()])
                    else:
                        new_args[class_key] = None

                self._trainer = builder.build_trainer(data_loader_class,
                                                      network_class,
                                                      trainer_class,
                                                      new_args['dd'],
                                                      dataset_dirs_class,
                                                      new_args['ds'],
                                                      dataset_spec_class,
                                                      new_args['dls'],
                                                      data_loader_spec_class,
                                                      new_args['ns'],
                                                      network_spec_class,
                                                      new_args['ts'],
                                                      trainer_spec_class)
                self._trainer.setup()

            def step(self):
                self._trainer_global_step = self._trainer.train(self._trainer_global_step,
                                                                self._trainer_global_step +
                                                                self._train_steps_per_tune_step)
                self.timestep += 1
                return {'cost': ivy.to_numpy(self._trainer.moving_average_loss)}

            def save_checkpoint(self, checkpoint_dir):
                path = os.path.join(checkpoint_dir, 'checkpoint')
                with open(path, "w") as f:
                    f.write(json.dumps({"timestep": self.timestep}))
                self._trainer.save(checkpoint_dir)
                return path

            def load_checkpoint(self, checkpoint_path):
                with open(checkpoint_path) as f:
                    self.timestep = json.loads(f.read())["timestep"]
                self._trainer.restore(checkpoint_path, self._trainer_global_step)

        # Run this trainable class #
        # -------------------------#

        ahb = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="cost",
            mode="min",
            grace_period=1,
            max_t=max_t)

        num_cpus = multiprocessing.cpu_count()
        config = _convert_tune_config(config)
        config['framework'] = ivy.get_framework_str()
        config['train_steps_per_tune_step'] = trainer_spec_args['train_steps_per_tune_step']
        ivy.unset_framework()

        reporter = CLIReporter(['cost'])

        tune.run(TuneTrainable,
                 progress_reporter=reporter,
                 name=name,
                 scheduler=ahb,
                 stop={"training_iteration": max_t},
                 num_samples=num_samples,
                 resources_per_trial={
                     "cpu": num_cpus,
                     "gpu": num_gpus
                 },
                 config=config)
        ivy.set_framework(config['framework'])
