# global
import ivy
import logging
import argparse

# local
import ivy_builder.builder as trainer_builder
from ivy_builder.abstract.network import Network
from ivy_builder.abstract.trainer import Trainer
from ivy_builder.abstract.data_loader import DataLoader


# noinspection PyAttributeOutsideInit
class ExampleNetwork(Network):

    def __init__(self, network_spec):
        super().__init__(network_spec)

    def _build(self, *args, **kwargs):
        self._l1 = ivy.Linear(1, 1)

    def _forward(self, x):
        return self._l1(x)


class ExampleDataLoader(DataLoader):

    def __init__(self, data_loader_spec):
        super().__init__(data_loader_spec)

    def get_next_batch(self, dataset_key=None):
        return ivy.Container(x=ivy.array([[1.]]*self._spec.batch_size, dev_str=self._spec.dev_strs[0]),
                             target=ivy.array([[0.]]*self._spec.batch_size, dev_str=self._spec.dev_strs[0]))

    def get_first_batch(self, dataset_key=None):
        return ivy.Container(x=ivy.array([[1.]]*self._spec.batch_size, dev_str=self._spec.dev_strs[0]),
                             target=ivy.array([[0.]]*self._spec.batch_size, dev_str=self._spec.dev_strs[0]))


class ExampleTrainer(Trainer):

    def __init__(self, trainer_spec):
        super().__init__(trainer_spec)
        self._sgd_optimizer = ivy.SGD(self._spec.initial_learning_rate)

    def _compute_cost(self, network, batch, dev_str, v=None):
        network_output = network(batch.x, v=v)
        return ivy.reduce_mean((batch.target - network_output)**2)[0]

    def _learning_rate_func(self, global_step):
        return self._spec.initial_learning_rate

    def _write_scalar_summaries(self, data_loader, network, training_batch, global_step):
        logging.info('step ' + str(self._global_step) + ': cost = ' + str(ivy.to_numpy(self._total_cost)))

    def _write_image_summaries(self, data_loader, network, training_batch, global_step):
        pass

    @property
    def _optimizer(self):
        return self._sgd_optimizer


# noinspection PyShadowingBuiltins
def main(seed=0, compile_mode=False, dev_strs=None):
    ivy.seed(seed)
    data_loader_spec_args = {'batch_size': 2, 'dev_strs': [ivy.default(lambda: dev_strs[0], None, True)]}
    trainer_spec_args = {'total_iterations': 10, 'ld_chkpt': False, 'log_freq': 1, 'initial_learning_rate': 0.1,
                         'compile_mode': compile_mode, 'dev_strs': dev_strs}
    trainer = trainer_builder.build_trainer(ExampleDataLoader, ExampleNetwork, ExampleTrainer,
                                            data_loader_spec_args=data_loader_spec_args,
                                            trainer_spec_args=trainer_spec_args)
    trainer.setup()
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
