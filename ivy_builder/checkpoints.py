# global
import os
import ivy


class Checkpoint:

    def __init__(self, optimizer, net):
        self._optimizer = optimizer
        self._net = net

    # noinspection PyProtectedMember
    def restore(self, checkpoint_path):
        checkpoint = ivy.Container.from_disk_as_hdf5(checkpoint_path)
        loaded_v = checkpoint.network.map(lambda x, kc: ivy.variable(ivy.to_dev(x, self._net._dev_str)))
        if ivy.exists(self._net.v):
            # if build_mode is 'on_call', the network variables will not have been built yet
            assert (self._net.v.shapes == loaded_v.shapes).all_true(assert_is_bool=True)
        self._net.v = loaded_v
        self._optimizer.set_state(checkpoint.optimizer.map(lambda x, kc: ivy.to_dev(x, self._net.spec.dev_strs[0])))

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
        self._latest_checkpoint_fpath = None
        self._get_latest_checkpoint_fpath()

    def _get_latest_checkpoint_fpath(self):
        if not os.path.exists(self._directory):
            return
        contents = os.listdir(self._directory)
        if contents:
            contents.sort(key=lambda x: int(x.split('-')[-1].split('.hdf5')[0]))
            self._latest_checkpoint_fpath = os.path.join(self._directory, contents[-1])

    @property
    def latest_checkpoint_fpath(self):
        return self._latest_checkpoint_fpath

    def save(self, step):
        checkpoint = ivy.Container({'network': self._checkpoint.net.v,
                                    'optimizer': self._checkpoint.optimizer.state})
        self._latest_checkpoint_fpath = os.path.join(self._directory, 'chkpt-{}.hdf5'.format(step))
        checkpoint.to_disk_as_hdf5(self._latest_checkpoint_fpath)
