# global
import os
import ivy
import logging
import argparse
import numpy as np
from ivy_demo_utils.framework_utils import get_framework_from_str, choose_random_framework

# local
import ivy_builder.builder as builder
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.trainer import Trainer
from ivy_builder.specs import DataLoaderSpec
from ivy_builder.specs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs import NetworkSpec


# Custom Specification Classes #
# -----------------------------#

class ExampleDatasetDirs(DatasetDirs):

    def __init__(self):
        super().__init__()
        self._root_dir = '/some/dataset/dir'
        self._vector_dir = os.path.join(self._root_dir, 'vectors')
        self._image_dir = os.path.join(self._root_dir, 'images')

    # Getters #
    # --------#

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def vector_dir(self):
        return self._vector_dir

    @property
    def image_dir(self):
        return self._image_dir


class ExampleDatasetSpec(DatasetSpec):

    def __init__(self, dirs, num_examples, vector_dim, image_dims):
        super().__init__(dirs)
        self._num_examples = num_examples
        self._vector_dim = vector_dim
        self._image_dims = image_dims

    # Getters #
    # --------#

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def vector_dim(self):
        return self._vector_dim

    @property
    def image_dims(self):
        return self._image_dims


class ExampleDataLoaderSpec(DataLoaderSpec):

    def __init__(self, dataset_spec, network, batch_size, shuffle):
        super().__init__(dataset_spec, network=network, batch_size=batch_size)
        self._shuffle = shuffle

    # Getters #
    # --------#

    @property
    def shuffle(self, **kwargs):
        return self._shuffle


class ExampleNetworkSpec(NetworkSpec):

    def __init__(self, device, num_layers):
        super().__init__(device)
        self._num_layers = num_layers

    # Getters #
    # --------#

    @property
    def num_layers(self):
        return self._num_layers


# Custom Data Loader #
# -------------------#

class ExampleDataLoader(DataLoader):

    def __init__(self, data_loader_spec):
        super().__init__(data_loader_spec)

        # dataset size
        self._num_examples = self._spec.dataset_spec.num_examples

        # counter
        self._i = 0

        # load vector data
        vector_dim = self._spec.dataset_spec.vector_dim
        vector_dir = self._spec.dataset_spec.dirs.vector_dir
        print('loading vector data from ' + vector_dir + '...')
        self._targets = ivy.zeros((self._num_examples, vector_dim, 1))

        # load image data
        image_dims = self._spec.dataset_spec.image_dims
        image_dir = self._spec.dataset_spec.dirs.image_dir
        print('loading image data from ' + image_dir + '...')
        self._input = ivy.ones((self._num_examples, image_dims[0], image_dims[1], 3))

        self._training_data = {'targets': self._targets, 'input': self._input}
        self._validation_data = {'targets': self._targets, 'input': self._input}
        self._data = {'training': self._training_data, 'validation': self._validation_data}

    def get_next_batch(self, dataset_key):
        data = self._data[dataset_key]
        if self._spec.shuffle:
            self._i = np.random.randint(0, self._num_examples)
        else:
            self._i = (self._i + 1) % self._num_examples
        return data['input'][self._i], data['targets'][self._i]

    def get_next_training_batch(self):
        return self.get_next_batch('training')


# Custom Network #
# ---------------#

class ExampleNetwork(Network, ivy.Module):

    def __init__(self, network_spec):
        num_layers = network_spec.num_layers
        self._layers = list()
        for i in range(num_layers):
            self._layers.append(ivy.Linear(3, 1))
        super().__init__(network_spec)

    # noinspection PyMethodOverriding
    def _forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return ivy.expand_dims(x, 0)


# Custom Trainer #
# ---------------#


# noinspection PyCallingNonCallable
class ExampleTrainer(Trainer):

    def __init__(self, trainer_spec):
        super().__init__(trainer_spec)
        self._sgd_optimizer = ivy.SGD(self._spec.initial_learning_rate)

    def _compute_cost(self, batch, v):
        target = batch[1]
        network_output = self._spec.network(batch[0], v=v)
        return ivy.reduce_mean((network_output - target) ** 2)[0]

    def _learning_rate_func(self, global_step):
        if global_step < self._spec.total_iterations/2:
            return self._spec.initial_learning_rate
        return self._spec.initial_learning_rate/2

    def _write_scalar_summaries(self, data_loader, network, training_batch, global_step):
        logging.info('step ' + str(self._global_step) + ': cost = ' + str(ivy.to_numpy(self._total_cost)))

    def _write_image_summaries(self, data_loader, network, training_batch, global_step):
        pass

    @property
    def _optimizer(self):
        return self._sgd_optimizer


# Training Job #
# -------------#

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # dataset dirs specification
    dataset_dirs_args = dict()

    # dataset specification
    dataset_spec_filepath = os.path.join(current_dir, 'json_specs', 'dataset_spec.json.example')
    dataset_spec_args = builder.parse_json_to_dict(dataset_spec_filepath)

    # data loader specification
    data_loader_spec_filepath = os.path.join(current_dir, 'json_specs', 'data_loader_spec.json.example')
    data_loader_spec_args = builder.parse_json_to_dict(data_loader_spec_filepath)

    # network specification
    network_spec_filepath = os.path.join(current_dir, 'json_specs', 'network_spec.json.example')
    network_spec_args = builder.parse_json_to_dict(network_spec_filepath)

    # trainer specification
    trainer_spec_filepath = os.path.join(current_dir, 'json_specs', 'trainer_spec.json.example')
    trainer_spec_args = builder.parse_json_to_dict(trainer_spec_filepath)

    # In all above cases, the user could override the loaded json file dicts with command line args if so desired
    # before then passing into the TrainingJob for specification class construction, which are all then read-only

    trainer = builder.build_trainer(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                    dataset_dirs_args, ExampleDatasetDirs, dataset_spec_args,
                                    ExampleDatasetSpec, data_loader_spec_args, ExampleDataLoaderSpec,
                                    network_spec_args, ExampleNetworkSpec, trainer_spec_args)
    trainer.setup()
    print("Finished complete example!")
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parsed_args = parser.parse_args()
    f = None if parsed_args.framework is None else get_framework_from_str(parsed_args.framework)
    f = choose_random_framework() if f is None else f
    ivy.set_framework(f)
    main()
    ivy.unset_framework()
