# global
import ivy
import logging
import argparse
from ivy_demo_utils.framework_utils import get_framework_from_str, choose_random_framework

# local
import ivy_builder.builder as trainer_builder
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.trainer import Trainer
from ivy_builder.abstract.data_loader import DataLoader


class ExampleNetwork(Network, ivy.Module):

    def __init__(self, network_spec):
        self._l1 = ivy.Linear(1, 1)
        super().__init__(network_spec)

    def _forward(self, x):
        return self._l1(x)


class ExampleDataLoader(DataLoader):

    def __init__(self, data_loader_spec):
        super().__init__(data_loader_spec)

    def get_next_batch(self, dataset_key):
        return ivy.array([[1.]])

    def get_next_training_batch(self):
        return ivy.array([[1.]])


class ExampleTrainer(Trainer):

    def __init__(self, trainer_spec):
        super().__init__(trainer_spec)
        self._sgd_optimizer = ivy.SGD(self._spec.initial_learning_rate)

    def _compute_cost(self, batch, v):
        network_output = self._spec.network(batch, v=v)
        target = ivy.array([[0.]])
        return ivy.reduce_mean((network_output - target)**2)[0]

    def _learning_rate_func(self, global_step):
        return self._spec.initial_learning_rate

    def _write_scalar_summaries(self, data_loader, network, training_batch, global_step):
        logging.info('step ' + str(self._global_step) + ': cost = ' + str(ivy.to_numpy(self._total_cost)))

    def _write_image_summaries(self, data_loader, network, training_batch, global_step):
        pass

    @property
    def _optimizer(self):
        return self._sgd_optimizer


def main(seed=0):
    ivy.seed(seed)
    data_loader_spec_args = {'batch_size': 1}
    trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'log_freq': 1, 'initial_learning_rate': 0.1}
    trainer = trainer_builder.build_trainer(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                            data_loader_spec_args=data_loader_spec_args,
                                            trainer_spec_args=trainer_spec_args)
    trainer.setup()
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
