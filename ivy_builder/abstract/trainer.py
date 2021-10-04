# global
import os
import ivy
try:
    import git
except ModuleNotFoundError:
    git = None
import abc
import time
import shutil
import psutil
import pathlib
import logging
import datetime
import nvidia_smi
import numpy as np
from datetime import datetime

# local
from ivy_builder.abstract.network import Network
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.builder import spec_to_dict, trainer_to_spec_args_dict, save_dict_as_json
from ivy_builder.checkpoints import Checkpoint, CheckpointManager

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

MIN_DENOMINATOR = 1e-12


def _get_valid_filepath(base_dir, base_filename, file_type):
    i = 0
    while True:
        filename = base_filename + str(i) if i != 0 else base_filename
        filepath = os.path.join(base_dir, filename + file_type)
        if os.path.exists(filepath):
            i += 1
            continue
        return filepath


class Trainer:

    def __init__(self, spec: TrainerSpec) -> None:

        # specification
        self._spec = spec

        # uninitialized variables
        if spec.starting_iteration is not None:
            self._starting_iteration = spec.starting_iteration
        else:
            self._starting_iteration = 0
        self._total_iterations = None

        # trainer variables
        self._global_step = 0
        self._moving_average_loss = 0

        # set seed
        np.random.seed(self._spec.seed)
        ivy.seed(self._spec.seed)

        # uninitialized variables
        self._chkpt = None
        self._chkpt_manager = None

        # summary writer
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            SummaryWriter = None
        if SummaryWriter is not None:
            self._writer = SummaryWriter(os.path.join(self._spec.log_dir, 'tnsrbrd'))
        else:
            self._writer = None

        # profiler
        self._profiling = self._spec.steps_to_profile > 0
        if self._profiling:
            self._profiler = ivy.Profiler(self._spec.log_dir)
        else:
            self._profiler = None

        # gpu memory logging
        # noinspection PyBroadException
        try:
            nvidia_smi.nvmlInit()
            self._gpu_handles = [nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                                 for i in range(nvidia_smi.nvmlDeviceGetCount())]
        except Exception:
            self._gpu_handles = list()

        # timing
        self._start_time = time.perf_counter()

        # batch
        self._training_batch = None

        # network
        self._network = self._spec.network
        self._net_spec = self._network.spec
        self._partial_grad_updates = bool(self._net_spec.v_keychains)

        # compile
        if self._spec.compile:
            self._train_step_from_batch = ivy.compile_fn(self._train_step_from_batch)

        # multi-dev
        if isinstance(self._spec.dev_strs, list) and len(self._spec.dev_strs) > 1:
            if self._network.built:
                raise Exception('Network must use either explicit or on_call build modes if training on multiple'
                                'devices, but the network was already built using on_init method.')
            self._dev_mapper = ivy.DevMapperMultiProc(self._execute_with_gradients, self._spec.dev_strs, self._network)
        else:
            self._dev_mapper = None

    # Abstract #
    # ---------#

    # Private Methods #

    @abc.abstractmethod
    def _compute_cost(self, network: ivy.Module, batch: ivy.Array, v: ivy.Container = None) -> ivy.Array:
        """
        compute training cost from input batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _learning_rate_func(self, global_step: ivy.Variable) -> ivy.Array:
        """
        compute learning rate, given global step
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_scalar_summaries(self, data_loader: DataLoader, network: Network, training_batch: ivy.Array,
                                global_step: ivy.Variable) -> None:
        """
        write scalar summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_image_summaries(self, data_loader: DataLoader, network: Network, training_batch: ivy.Array,
                               global_step: ivy.Variable) -> None:
        """
        write image summaries to disk, ready for tensorboard viewing
        """
        raise NotImplementedError

    def _pre_init(self) -> None:
        """
        Initialize the trainer, called before checkpoints are loaded and the network is built. Optional function.
        """
        pass

    def _post_init(self) -> None:
        """
        Initialize the trainer, called after checkpoints are loaded and the network is built. Optional function.
        """
        pass

    # Getters #

    @property
    @abc.abstractmethod
    def _optimizer(self):
        raise NotImplementedError

    @_optimizer.setter
    def _optimizer(self, value):
        self._optimizer = value

    # Initialization #
    # ---------------#

    def _init_checkpoint_manager(self):
        pathlib.Path(os.path.join(self._spec.log_dir, 'chkpts')).mkdir(parents=True, exist_ok=True)
        self._chkpt = Checkpoint(optimizer=self._optimizer, net=self._network)
        self._chkpt_manager = CheckpointManager(self._chkpt, os.path.join(self._spec.log_dir, 'chkpts'), 20,
                                                step_counter=self._global_step)

    def _log_scalars(self):
        if ivy.exists(self._writer):
            if self._spec.log_time:
                self._writer.add_scalar('time between logs', time.perf_counter() - self._start_time, self._global_step)
            if self._spec.log_learning_rate:
                self._writer.add_scalar('learning rate', self._learning_rate, self._global_step)
        self._write_scalar_summaries(self._spec.data_loader, self._network, self._training_batch,
                                     self._global_step)
        self._start_time = time.perf_counter()

    def _log_nested(self, nest, global_step, name_hierarchy, spec):
        if not ivy.exists(self._writer):
            raise Exception('torch must be installed in order to use the file writer for tensorboard logging.')
        if 'global_vector_norm' in spec:
            self._writer.add_scalar(name_hierarchy + '/global vector norm',
                                    ivy.to_scalar(ivy.to_native(nest.vector_norm(global_norm=True))), global_step)
        for k, v in nest.items():
            new_name_hierarchy = name_hierarchy + '/' + k
            if isinstance(v, dict):
                self._log_nested(v, global_step, new_name_hierarchy, spec)
            else:
                if 'mean' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/mean',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_mean(v))), global_step)
                if 'abs_mean' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/abs mean',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_mean(ivy.abs(v)))), global_step)
                if 'var' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/var',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_var(v))), global_step)
                if 'abs_var' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/abs var',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_var(ivy.abs(v)))), global_step)
                if 'min' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/min',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_min(v))), global_step)
                if 'abs_min' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/abs min',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_min(ivy.abs(v)))), global_step)
                if 'max' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/max',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_max(v))), global_step)
                if 'abs_max' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/abs max',
                                            ivy.to_scalar(ivy.to_native(ivy.reduce_max(ivy.abs(v)))), global_step)
                if 'vector_norm' in spec:
                    self._writer.add_scalar(new_name_hierarchy + '/vector norm',
                                            ivy.to_scalar(ivy.to_native(ivy.vector_norm(v))), global_step)

    def _log_gradients(self, grads, global_step):
        self._log_nested(grads, global_step, 'gradients', self._spec.log_gradients)

    def _log_variables(self, v, global_step):
        self._log_nested(v, global_step, 'variables', self._spec.log_variables)

    def _log_optimizer_state(self, optimizer_state, global_step):
        self._log_nested(optimizer_state, global_step, 'optimizer_state', self._spec.log_optimizer_state)

    def _log_memory(self, global_step):
        if not ivy.exists(self._writer):
            raise Exception('torch must be installed in order to use the file writer for tensorboard logging.')
        vm = psutil.virtual_memory()
        self._writer.add_scalar('memory/RAM/global/percent_used', (1-(vm.available/vm.total))*100, global_step)
        this_process = psutil.Process(os.getpid())
        self._writer.add_scalar('memory/RAM/local/percent_used',
                                (this_process.memory_info().rss/vm.total)*100, global_step)
        for i, handle in enumerate(self._gpu_handles):
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self._writer.add_scalar('memory/GPU_{}/global/percent_used'.format(i),
                                    (info.used/info.total)*100, global_step)

    def _save(self):
        self._chkpt_manager.save(self._global_step)
        logging.info('network checkpoint saved @ step ' + str(self._global_step))

    def _save_spec_to_disk(self):

        # remove/create log dir
        if self._spec.overwrite_log_dir:
            shutil.rmtree(self._spec.log_dir, ignore_errors=True)
        os.makedirs(self._spec.log_dir, exist_ok=True)

        # create directory
        spec_dir = os.path.join(self._spec.log_dir, 'spec')
        os.makedirs(spec_dir, exist_ok=True)

        # write spec json
        complete_spec_filepath = _get_valid_filepath(spec_dir, 'complete_spec', '.json')
        spec_dict = spec_to_dict(self._spec)
        save_dict_as_json(spec_dict, complete_spec_filepath)

        # write spec args json
        complete_spec_args_filepath = _get_valid_filepath(spec_dir, 'complete_spec_args', '.json')
        spec_args_dict = trainer_to_spec_args_dict(self)
        save_dict_as_json(spec_args_dict, complete_spec_args_filepath)

    def _save_info_to_disk(self):
        info_dir = os.path.join(self._spec.log_dir, 'info')
        os.makedirs(info_dir, exist_ok=True)
        info_filepath = _get_valid_filepath(info_dir, 'info', '.txt')
        if not ivy.exists(git):
            logging.warning('no gitpython installation found, not saving git commit hash to disk. '
                            'To install gitpython, run pip install gitpython.')
            return
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
        except (git.exc.InvalidGitRepositoryError, ValueError):
            sha = 'NOT A GIT REPO'
        with open(info_filepath, 'w+') as info_file:
            info_file.writelines(['time of execution:\n',
                                  str(datetime.now()) + '\n\n',
                                  'git commit hash at time of execution:\n',
                                  sha + '\n'])

    def _initialize_model(self, checkpoint_path=None):
        self._pre_init()
        if self._net_spec.build_mode == 'explicit':
            self._network.build()
        self._compute_cost(self._network, self._spec.data_loader.get_first_batch())  # for on_call builds
        if self._spec.save_spec:
            self._save_spec_to_disk()
        self._save_info_to_disk()
        self._init_checkpoint_manager()
        if not checkpoint_path:
            checkpoint_path = self._chkpt_manager.latest_checkpoint_fpath
        if self._spec.ld_chkpt is True and not ivy.exists(checkpoint_path):
            raise Exception('Unable to load checkpoint, no checkpoint files found.')
        elif self._spec.ld_chkpt in [True, 'try'] and ivy.exists(checkpoint_path):
            self._chkpt.restore(checkpoint_path)
            logging.info('loaded checkpoints from {}'.format(checkpoint_path))
            starting_iteration = int(checkpoint_path.split('-')[-1].split('.')[0])
            logging.info('#--------------#\n# MODEL LOADED #\n#--------------#')
            self._post_init()
            if ivy.exists(self._spec.starting_iteration):
                assert starting_iteration == self._spec.starting_iteration
            return starting_iteration
        else:
            logging.info('#-------------#\n# MODEL BUILT #\n#-------------#')
        self._global_step = self._spec.starting_iteration
        self._post_init()
        return ivy.default(self._spec.starting_iteration, 0)

    # Training #
    # ---------#

    def _execute_with_gradients(self, network, batch, network_v):
        cost, gradients = ivy.execute_with_gradients(
            lambda v: self._compute_cost(network, batch, v=network_v.set_at_key_chains(v)),
            network_v.at_key_chains(self._net_spec.v_keychains, ignore_none=True) if
            self._net_spec.keep_v_keychains else
            network_v.prune_key_chains(self._net_spec.v_keychains, ignore_none=True))
        return cost, gradients

    def _execute_with_gradients_multi_dev(self, network, batch):
        if ivy.exists(self._dev_mapper):
            if not isinstance(batch, ivy.MultiDevContainer):
                batch = batch.to_multi_dev(self._spec.dev_strs)
            return self._dev_mapper.map(batch, network.v.clone(self._spec.dev_strs))
        return self._execute_with_gradients(network, batch, network.v)

    def _train_step_from_batch(self, batch):
        cost, self._gradients = self._execute_with_gradients_multi_dev(self._network, batch)
        if 'max_grad_val' in self._spec:
            grads = self._gradients.clip(-self._spec.max_grad_val, self._spec.max_grad_val)
        if 'max_grad_vector_norm' in self._spec:
            ratio = self._spec.max_grad_vector_norm/(self._gradients.vector_norm(global_norm=True) + MIN_DENOMINATOR)
            if ratio < 1:
                self._gradients = self._gradients * ratio
        self._moving_average_loss = (cost + self._global_step * self._moving_average_loss) / (self._global_step + 1)
        new_v = self._optimizer.step(self._network.v, self._gradients, ignore_missing=self._partial_grad_updates)
        return batch, cost, new_v

    def _train_step(self, with_output=False):
        batch, cost, new_v = self._train_step_from_batch(self._spec.data_loader.get_next_batch())
        if self._partial_grad_updates:
            self._network.v.set_at_key_chains(new_v)
        else:
            self._network.v = new_v
        if with_output:
            return batch, cost
        return cost

    def _data_load_and_train_step(self, vis_mode, log_scalars_on_this_it, log_viz_on_this_it):
        if vis_mode:
            self._training_batch = self._spec.data_loader.get_next_batch()
        else:
            if log_scalars_on_this_it or log_viz_on_this_it:
                self._training_batch, self._total_cost = \
                    self._train_step(with_output=True)
            else:
                self._total_cost = self._train_step()

    def _train(self, vis_mode=False, starting_iteration=None, total_iterations=None):

        self._starting_iteration = ivy.default(starting_iteration, self._starting_iteration)
        self._total_iterations = ivy.default(total_iterations, self._spec.total_iterations)

        self._global_step = self._starting_iteration
        self._learning_rate = self._learning_rate_func(self._global_step)

        if self._starting_iteration == self._total_iterations:
            return self._starting_iteration

        if vis_mode:
            vis_freq = 1
        else:
            vis_freq = self._spec.vis_freq

        local_counter = 0

        while self._global_step < self._total_iterations or self._total_iterations == -1:

            if self._profiling and local_counter == self._spec.profile_start_step:
                self._profiler.start()

            final_step = self._global_step == self._total_iterations - 1
            log_scalars = (self._global_step % self._spec.log_freq == 0 or (final_step and self._spec.log_at_end)) \
                          and self._spec.log_freq > 0 and not vis_mode
            log_viz = (self._global_step % vis_freq == 0 or (final_step and self._spec.vis_at_end)) \
                      and self._spec.vis_freq > 0
            save = (self._global_step % self._spec.save_freq == 0 or (final_step and self._spec.save_at_end)) \
                   and self._spec.save_freq > 0 and not vis_mode

            self._data_load_and_train_step(vis_mode, log_scalars, log_viz)

            if log_scalars:
                self._log_scalars()
            if log_viz or vis_mode:
                self._write_image_summaries(self._spec.data_loader, self._network, self._training_batch,
                                            self._global_step)
            if save:
                self._save()

            self._global_step += 1
            local_counter += 1
            self._learning_rate = self._learning_rate_func(self._global_step)

            if vis_mode:
                input('press enter to visualise another example')

            if self._profiling and local_counter == self._spec.profile_start_step + self._spec.steps_to_profile:
                self._profiler.stop()

        return self._global_step

    # Public Methods #
    # ---------------#

    def save(self, checkpoint_path: str) -> None:
        """
        save the network weights and optimizer state in checkpoint file
        :param checkpoint_path: path of the checkpoint file for saving the weights and optimizer state
        """
        checkpoint = ivy.Container({'network': self._network.v,
                                    'optimizer': self._optimizer.state})
        os.makedirs('/'.join(checkpoint_path.split('/')[:-1]), exist_ok=True)
        checkpoint.to_disk_as_hdf5(checkpoint_path)

    def restore(self, checkpoint_path: str, global_step: int = None) -> None:
        """
        restore the network weights from checkpoint file
        :param checkpoint_path: path of the checkpoint file for loading the weights
        :param global_step: training step to start at for continued training
        """
        self._chkpt.restore(checkpoint_path)
        if global_step is not None:
            self._global_step = global_step

    def setup(self) -> None:
        """
        setup the trainer, ready for training
        """
        self._starting_iteration = self._initialize_model()

    def train(self, starting_iteration: int = None, total_iterations: int = None) -> None:
        """
        run the trainer, returning the iteration step reached
        """
        self._start_time = time.perf_counter()
        return self._train(False, starting_iteration, total_iterations)

    def visualize(self) -> None:
        """
        run the trainer, but without weight optimization. Only used for tensorboard visualization
        """
        self._train(True)

    def close(self) -> None:
        """
        Close this trainer, and destroy all child objects or processes which may not be garbage collected.
        """
        self._spec.data_loader.close()

    # Getters #
    # --------#

    @property
    def learning_rate(self):
        return self._learning_rate_func(self._global_step)

    @property
    def spec(self):
        return self._spec

    @property
    def moving_average_loss(self):
        return self._moving_average_loss
