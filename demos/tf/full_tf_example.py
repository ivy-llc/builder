# global
import os

import numpy as np
import tensorflow as tf

# local
import ivy_builder.builder as builder
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.trainers.tf_trainer import TFTrainer
from ivy_builder.specs import DataLoaderSpec
from ivy_builder.specs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs import NetworkSpec
from ivy_builder.specs.trainer_spec import TrainerSpec


# Custom Specification Classes #
# -----------------------------#

class ExampleTFDatasetDirs(DatasetDirs):

    def __init__(self):
        super().__init__()
        self.__root_dir = '/some/dataset/dir'
        self.__vector_dir = os.path.join(self.__root_dir, 'vectors')
        self.__image_dir = os.path.join(self.__root_dir, 'images')

    # Getters #
    # --------#

    @property
    def root_dir(self):
        return self.__root_dir

    @property
    def vector_dir(self):
        return self.__vector_dir

    @property
    def image_dir(self):
        return self.__image_dir


class ExampleTFDatasetSpec(DatasetSpec):

    def __init__(self, dirs, num_examples, vector_dim, image_dims):
        super().__init__(dirs)
        self.__num_examples = num_examples
        self.__vector_dim = vector_dim
        self.__image_dims = image_dims

    # Getters #
    # --------#

    @property
    def num_examples(self):
        return self.__num_examples

    @property
    def vector_dim(self):
        return self.__vector_dim

    @property
    def image_dims(self):
        return self.__image_dims


class ExampleTFDataLoaderSpec(DataLoaderSpec):

    def __init__(self, dataset_spec, network, batch_size, shuffle):
        super().__init__(dataset_spec, network, batch_size)
        self.__shuffle = shuffle

    # Getters #
    # --------#

    @property
    def shuffle(self, **kwargs):
        return self.__shuffle


class ExampleTFNetworkSpec(NetworkSpec):

    def __init__(self, num_layers):
        super().__init__()
        self.__num_layers = num_layers

    # Getters #
    # --------#

    @property
    def num_layers(self):
        return self.__num_layers


class ExampleTFTrainerSpec(TrainerSpec):

    def __init__(self, **kwargs):
        child_args = dict()
        for arg_key in ['learning_decrement', 'learning_decrement_rate', 'staircase']:
            child_args[arg_key] = kwargs[arg_key]
            del kwargs[arg_key]
        super().__init__(**kwargs)
        self.__learning_decrement = child_args['learning_decrement']
        self.__learning_decrement_rate = child_args['learning_decrement_rate']
        self.__staircase = child_args['staircase']

    # Getters #
    # --------#

    @property
    def learning_decrement(self):
        return self.__learning_decrement

    @property
    def learning_decrement_rate(self):
        return self.__learning_decrement_rate

    @property
    def staircase(self):
        return self.__staircase


# Custom Data Loader #
# -------------------#

class ExampleTFDataLoader(DataLoader):

    def __init__(self, data_loader_spec):
        super().__init__(data_loader_spec)

        # dataset size
        self.__num_examples = self._spec.dataset_spec.num_examples

        # counter
        self.__i = 0

        # load vector data
        vector_dim = self._spec.dataset_spec.vector_dim
        vector_dir = self._spec.dataset_spec.dirs.vector_dir
        print('loading vector data from ' + vector_dir + '...')
        self.__targets = tf.ones((self.__num_examples, vector_dim, 1))

        # load image data
        image_dims = self._spec.dataset_spec.image_dims
        image_dir = self._spec.dataset_spec.dirs.image_dir
        print('loading image data from ' + image_dir + '...')
        self.__input = tf.ones((self.__num_examples, image_dims[0], image_dims[1], 3))

        self.__training_data = {'targets': self.__targets, 'input': self.__input}
        self.__validation_data = {'targets': self.__targets, 'input': self.__input}
        self.__data = {'training': self.__training_data, 'validation': self.__validation_data}

    def get_next_batch(self, dataset_key):
        data = self.__data[dataset_key]
        if self._spec.shuffle:
            self.__i = np.random.uniform(0, self.__num_examples, 1)[0]
        else:
            self.__i = (self.__i + 1) % self.__num_examples
        index = tf.constant(self.__i, dtype=tf.int32)
        return data['input'][index], data['targets'][index]

    def get_next_training_batch(self):
        return self.get_next_batch('training')


# Custom Network #
# ---------------#

class ExampleTFNetwork(Network, tf.keras.Model):

    def __init__(self, network_spec):
        super().__init__(network_spec)
        num_layers = self._spec.num_layers
        self._model = tf.keras.Sequential()
        for i in range(num_layers):
            self._model.add(tf.keras.layers.Dense(3))

    # noinspection PyMethodOverriding
    def call(self, x):
        return tf.expand_dims(self._model(x), 0)

    def get_config(self):
        pass


# Custom Trainer #
# ---------------#


class ExampleTFTrainer(TFTrainer):

    def __init__(self, trainer_spec):
        super().__init__(trainer_spec)
        self._learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            trainer_spec.initial_learning_rate, trainer_spec.learning_decrement_rate,
            trainer_spec.learning_decrement, staircase=trainer_spec.staircase)
        self._adam_optimizer = tf.keras.optimizers.Adam(self._learning_rate)

        self._validation_scalar_writer =\
            tf.summary.create_file_writer(os.path.join(self._spec.log_dir, 'tnsrbrd', 'scalars', 'validation'))
        self._validation_image_writer =\
            tf.summary.create_file_writer(os.path.join(self._spec.log_dir, 'tnsrbrd', 'images', 'validation'))

    def _compute_cost(self, batch, tape):
        target = batch[1]
        network_output = self._spec.network(batch[0])
        return tf.square(network_output - target)

    def _learning_rate_func(self, global_step):
        return self._learning_rate_schedule(global_step)

    def _write_scalar_summaries(self, data_loader, network, training_batch, training_summary_writer, global_step):

        print('trained step ' + str(global_step.numpy()))

        # training #
        network_output = network.call(training_batch[0])
        mean_network_output = tf.reduce_mean(network_output)
        with training_summary_writer.as_default():
            tf.summary.scalar('mean_network_output', mean_network_output, step=global_step)
        if not self._spec.log_validation:
            return

        # validation #
        validation_batch = data_loader.get_next_batch('validation')
        network_output = network.call(validation_batch[0])
        mean_network_output = tf.reduce_mean(network_output)
        with self._validation_scalar_writer.as_default():
            tf.summary.scalar('mean_network_output', mean_network_output, step=global_step)

    def _write_image_summaries(self, data_loader, network, training_batch, training_summary_writer, global_step):

        # training #
        network_output = self._spec.network.call(training_batch[0], )
        with training_summary_writer.as_default():
            tf.summary.image('network_output', network_output, step=global_step)
        if not self._spec.log_validation:
            return

        # validation #
        validation_batch = self._spec.data_loader.get_next_batch('validation')
        log_tensors = self._spec.network.call(validation_batch[0], )
        with self._validation_image_writer.as_default():
            tf.summary.image('network_output', log_tensors, step=global_step)

    @property
    def _optimizer(self):
        return self._adam_optimizer


# Training Job #
# -------------#

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # dataset dirs specification
    dataset_dirs_args = dict()

    # dataset specification
    dataset_spec_filepath = os.path.join(current_dir, '../json_specs', 'dataset_spec.json.example')
    dataset_spec_args = builder.parse_json_to_dict(dataset_spec_filepath)

    # data loader specification
    data_loader_spec_filepath = os.path.join(current_dir, '../json_specs', 'data_loader_spec.json.example')
    data_loader_spec_args = builder.parse_json_to_dict(data_loader_spec_filepath)

    # network specification
    network_spec_filepath = os.path.join(current_dir, '../json_specs', 'network_spec.json.example')
    network_spec_args = builder.parse_json_to_dict(network_spec_filepath)

    # trainer specification
    trainer_spec_filepath = os.path.join(current_dir, '../json_specs', 'trainer_spec.json.example')
    trainer_spec_args = builder.parse_json_to_dict(trainer_spec_filepath)

    # In all above cases, the user could override the loaded json file dicts with command line args if so desired
    # before then passing into the TrainingJob for specification class construction, which are all then read-only

    trainer = builder.build_trainer(ExampleTFDataLoader, ExampleTFNetwork, ExampleTFTrainer,
                                    dataset_dirs_args, ExampleTFDatasetDirs, dataset_spec_args,
                                    ExampleTFDatasetSpec, data_loader_spec_args, ExampleTFDataLoaderSpec,
                                    network_spec_args, ExampleTFNetworkSpec, trainer_spec_args,
                                    ExampleTFTrainerSpec)
    trainer.setup()
    print("Finished complete example!")
    trainer.train()


if __name__ == '__main__':
    main()
