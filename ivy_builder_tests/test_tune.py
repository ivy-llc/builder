# global
import os
import ray
import ivy
import json
import pytest
import random
from ray import tune
import tensorflow as tf
from ray.tune import CLIReporter
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler

# local
import ivy_tests.helpers as helpers
import ivy_builder.builder as builder
from ivy.core.container import Container
import ivy_builder_tests.helpers as builder_helpers
from demos.simple_example import ExampleDataLoader, ExampleNetwork, ExampleTrainer

ray.init()


def test_tune_alone(dev_str, call):

    if call is helpers.mx_call:
        # error processing isnan for mxnet
        pytest.skip()

    builder_helpers.remove_dirs()

    class TuneTrainable(tune.Trainable):

        def setup(self, _):
            print('EXPERIMENT_ID: {}'.format(self._experiment_id))
            self.timestep = 0

        def step(self):
            self.timestep += 1
            cost = ivy.tanh(ivy.array([float(self.timestep) / self.config.width]))
            cost *= self.config.height
            return {"cost": cost}

        def save_checkpoint(self, checkpoint_dir):
            path = os.path.join(checkpoint_dir, "checkpoint")
            with open(path, "w") as f:
                f.write(json.dumps({"timestep": self.timestep}))
            return path

        def load_checkpoint(self, checkpoint_path):
            with open(checkpoint_path) as f:
                self.timestep = json.loads(f.read())["timestep"]

    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="cost",
        mode="min",
        grace_period=5,
        max_t=5)

    reporter = CLIReporter(["cost"])

    tune.run(TuneTrainable,
             progress_reporter=reporter,
             name="asynchyperband_test",
             scheduler=ahb,
             stop={"training_iteration": 5},
             num_samples=5,
             resources_per_trial={
                 "cpu": 1,
                 "gpu": 0
             },
             config=Container({
                 "width": tune.sample_from(lambda spec: 10 + int(90 * random.random())),
                 "height": tune.sample_from(lambda spec: int(100 * random.random())),
             }))
    builder_helpers.remove_dirs()


def test_tune_integration(dev_str, call):

    if call is helpers.np_call:
        # Numpy does not support gradients
        pytest.skip()

    builder_helpers.remove_dirs()
    trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'log_freq': 1, 'train_steps_per_tune_step': 2}
    tuner = builder.build_tuner(ExampleDataLoader, ExampleNetwork, ExampleTrainer, trainer_spec_args=trainer_spec_args)
    num_gpus = 1 if tf.config.list_physical_devices('GPU') else 0
    tuner.run_tune('asynchyperband_test',
                   Container({'ts_initial_learning_rate':
                                  {'min': -6, 'max': -3, 'gaussian': False, 'exponential': True,
                                   'as_int': False}}), 2, 2, num_gpus)
    builder_helpers.remove_dirs()
