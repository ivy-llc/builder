# global
import os
import ivy
import logging
import argparse
import numpy as np

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

# noinspection PyAbstractClass
class ExampleDatasetDirs(DatasetDirs):

    def __init__(self):
        root_dir = '/some/dataset/dir'
        super().__init__(root_dir=root_dir,
                         vector_dir=os.path.join(root_dir, 'vectors'),
                         image_dir=os.path.join(root_dir, 'images'))


# noinspection PyAbstractClass
class ExampleDatasetSpec(DatasetSpec):

    def __init__(self, dirs, num_examples, vector_dim, image_dims):
        super().__init__(dirs,
                         num_examples=num_examples,
                         vector_dim=vector_dim,
                         image_dims=image_dims)


# noinspection PyAbstractClass
class ExampleDataLoaderSpec(DataLoaderSpec):

    def __init__(self, dataset_spec, batch_size, shuffle):
        super().__init__(dataset_spec,
                         batch_size=batch_size,
                         shuffle=shuffle)


# noinspection PyAbstractClass
class ExampleNetworkSpec(NetworkSpec):

    def __init__(self, dataset_spec, dev_strs, num_layers):
        super().__init__(dataset_spec,
                         dev_strs,
                         num_layers=num_layers)


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
        self._targets = ivy.zeros((self._num_examples, vector_dim, 1))

        # load image data
        image_dims = self._spec.dataset_spec.image_dims
        self._input = ivy.ones((self._num_examples, image_dims[0], image_dims[1], 3))

        self._training_data = ivy.Container(targets=self._targets, input=self._input)
        self._validation_data = ivy.Container(targets=self._targets, input=self._input)
        self._data = ivy.Container(training=self._training_data, validation=self._validation_data)

    def get_next_batch(self, dataset_key='training'):
        data = self._data[dataset_key]
        if self._spec.shuffle:
            self._i = np.random.randint(0, self._num_examples)
        else:
            self._i = (self._i + 1) % self._num_examples
        return ivy.Container(input=data.input[self._i], target=data.targets[self._i])

    def get_first_batch(self, dataset_key='training'):
        data = self._data[dataset_key]
        return ivy.Container(input=data.input[0], target=data.targets[0])


# Custom Network #
# ---------------#

# noinspection PyAttributeOutsideInit
class ExampleNetwork(Network, ivy.Module):

    def __init__(self, network_spec):
        super().__init__(network_spec)

    def _build(self, *args, **kwargs):
        self._layers = list()
        for i in range(self._spec.num_layers):
            self._layers.append(ivy.Linear(3, 1))

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

    def _compute_cost(self, network, batch, dev_str, v=None):
        network_output = network(batch.input, v=v)
        return ivy.reduce_mean((network_output - batch.target) ** 2)[0]

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

def main(compile_mode=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # dataset dirs specification
    dataset_dirs_args = dict()

    # dataset specification
    dataset_spec_filepath = os.path.join(current_dir, 'json_specs', 'dataset_spec.json.example')
    dataset_spec_args = builder.parse_json_to_cont(dataset_spec_filepath)

    # data loader specification
    data_loader_spec_filepath = os.path.join(current_dir, 'json_specs', 'data_loader_spec.json.example')
    data_loader_spec_args = builder.parse_json_to_cont(data_loader_spec_filepath)

    # network specification
    network_spec_filepath = os.path.join(current_dir, 'json_specs', 'network_spec.json.example')
    network_spec_args = builder.parse_json_to_cont(network_spec_filepath)

    # trainer specification
    trainer_spec_filepath = os.path.join(current_dir, 'json_specs', 'trainer_spec.json.example')
    trainer_spec_args = builder.parse_json_to_cont(trainer_spec_filepath)

    # In all above cases, the user could override the loaded json file dicts with command line args if so desired
    # before then passing into the TrainingJob for specification class construction, which are all then read-only

    trainer = builder.build_trainer(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                    dataset_dirs_args=dataset_dirs_args, dataset_dirs_class=ExampleDatasetDirs,
                                    dataset_spec_args=dataset_spec_args, dataset_spec_class=ExampleDatasetSpec,
                                    data_loader_spec_args=data_loader_spec_args,
                                    data_loader_spec_class=ExampleDataLoaderSpec, network_spec_args=network_spec_args,
                                    network_spec_class=ExampleNetworkSpec, trainer_spec_args=trainer_spec_args,
                                    spec_cont=ivy.Container({'trainer': {'compile_mode': compile_mode}}))
    trainer.setup()
    print("Finished complete example!")
    trainer.train()
    trainer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default=None,
                        help='which framework to use. Chooses a random framework if unspecified.')
    parsed_args = parser.parse_args()
    f = ivy.default(parsed_args.framework, ivy.choose_random_framework())
    ivy.set_framework(f)
    main()
    ivy.unset_framework()
