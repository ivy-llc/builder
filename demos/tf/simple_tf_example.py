# global
import tensorflow as tf

# local
import ivy_builder.builder as trainer_builder
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.trainers.tf_trainer import TFTrainer


class ExampleTFNetwork(Network, tf.keras.Model):

    def __init__(self, network_spec):
        super().__init__(network_spec)
        self._l1 = tf.keras.layers.Dense(5)

    # noinspection PyMethodOverriding
    def call(self, x):
        return self._l1(x)

    def get_config(self):
        pass


class ExampleTFDataLoader(DataLoader):

    def __init__(self, data_loader_spec):
        super().__init__(data_loader_spec)

    def get_next_batch(self, dataset_key):
        return tf.constant([[1.]])

    def get_next_training_batch(self):
        return tf.constant([[1.]])


class ExampleTFTrainer(TFTrainer):

    def __init__(self, trainer_spec):
        super().__init__(trainer_spec)
        self._adam_optimizer = tf.keras.optimizers.Adam(self._learning_rate)

    def _compute_cost(self, batch, tape):
        network_output = self._spec.network.call(batch)
        target = tf.constant([[0.]])
        return (network_output - target)**2

    def _learning_rate_func(self, global_step):
        return 1e-4

    def _write_scalar_summaries(self, data_loader, network, training_batch, training_summary_writer, global_step):
        print('trained step ' + str(global_step.numpy()))

    def _write_image_summaries(self, data_loader, network, training_batch, training_summary_writer, global_step):
        pass

    @property
    def _optimizer(self):
        return self._adam_optimizer


def main():
    data_loader_spec_args = {'batch_size': 1}
    trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'log_freq': 1}
    trainer = trainer_builder.build_trainer(ExampleTFDataLoader, ExampleTFNetwork, ExampleTFTrainer,
                                            data_loader_spec_args=data_loader_spec_args,
                                            trainer_spec_args=trainer_spec_args)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
