# global
import os
import ivy


class Checkpoint:

    def __init__(self, optimizer, net):
        self._optimizer = optimizer
        self._net = net

    def restore(self, checkpoint_path):
        checkpoint = ivy.Container.from_disk_as_hdf5(checkpoint_path)
        self._net.v = checkpoint.network.map(lambda x, kc: ivy.variable(ivy.to_dev(x, self._net.spec.device)))
        self._optimizer.set_state(checkpoint.optimizer.map(lambda x, kc: ivy.to_dev(x, self._net.spec.device)))

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def net(self):
        return self._net


class CheckpointManager:

    def __init__(self, checkpoint, directory, max_to_keep=20, step_counter=0):
        self._checkpoint = checkpoint
        self._directory = directory
        self._max_to_keep = max_to_keep
        self._step_counter = step_counter
        self._get_latest_checkpoint_fpath()

    def _get_latest_checkpoint_fpath(self):
        if os.path.exists(self._directory):
            contents = os.listdir(self._directory)
            if len(contents) == 0:
                self._latest_checkpoint_fpath = None
            else:
                contents.sort(key=lambda x: int(x.split('-')[-1].split('.hdf5')[0]))
                self._latest_checkpoint_fpath = os.path.join(self._directory, contents[-1])
        else:
            self._latest_checkpoint_fpath = None

    @property
    def latest_checkpoint_fpath(self):
        return self._latest_checkpoint_fpath

    def save(self, step):
        checkpoint = ivy.Container({'network': self._checkpoint.net.v,
                                    'optimizer': self._checkpoint.optimizer.state})
        self._latest_checkpoint_fpath = os.path.join(self._directory, 'chkpt-{}.hdf5'.format(step))
        checkpoint.to_disk_as_hdf5(self._latest_checkpoint_fpath)
