# local
from ivy_builder.abstract.trainer import Trainer


class EmptyTrainer(Trainer):

    def __init__(self, trainer_spec):
        super(EmptyTrainer, self).__init__(trainer_spec)

    def _compute_cost(self, batch, v):
        pass

    def _learning_rate_func(self, global_step):
        pass

    def _write_scalar_summaries(self, data_loader, network, training_batch, global_step):
        pass

    def _write_image_summaries(self, data_loader, network, training_batch, global_step):
        pass

    @property
    def _optimizer(self):
        return None
