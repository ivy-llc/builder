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
import pathlib
import logging
import datetime
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

        # timing
        self._start_time = time.perf_counter()

        # batch
        self._training_batch = None

        # network
        self._network = self._spec.network
        self._net_spec = self._network.spec
        self._partial_grad_updates = bool(self._net_spec.v_keychains)

        # multi-dev
        self._dev_str = ivy.default(lambda: self._spec.dev_strs[0], ivy.default_device(), True)
        if len(self._spec.dev_strs) > 1:
            if self._network.built:
                raise Exception('Network must use either explicit or on_call build modes if training on multiple'
                                'devices, but the network was already built using on_init method.')
            ret_fn = lambda ret: ivy.unify_iter(ret, self._spec.dev_strs[0], 'mean', transpose=True)
            dev_mapper = ivy.DevMapperMultiProc(
                self.__getattribute__(self._spec.dev_map_fn), ret_fn, self._spec.dev_strs,
                constant={'network': self._network})
            self._multi_dev = True
        else:
            dev_mapper = None
            self._multi_dev = False

        # device manager
        if (self._multi_dev and self._spec.tune_device_allocation) or self._spec.tune_splitting:
            self._dev_manager = ivy.DevManager(
                dev_mapper, self._spec.dev_strs, tune_dev_alloc=(self._multi_dev and self._spec.tune_device_allocation),
                tune_dev_splits=self._spec.tune_splitting)
        else:
            self._dev_manager = None

        # compilation
        self._compile_network_once_tuned = False
        self._compile_optimizer_once_tuned = False

    def __getstate__(self):
        # prevent already running processes from being pickled as sent to new processes
        state = self.__dict__.copy()
        state['_writer'] = None
        state['_profiler'] = None
        state['_gpu_handles'] = None
        state['_network'] = None
        state['_spec'] = None
        state['spec'] = None
        return state

    # Abstract #
    # ---------#

    # Private Methods #

    @abc.abstractmethod
    def _compute_cost(self, network: ivy.Module, batch: ivy.Array, dev_str: ivy.Device = None,
                      v: ivy.Container = None) -> ivy.Array:
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
        self._writer.add_scalar('memory/RAM/global/percent_used', ivy.percent_used_mem_on_dev('cpu'), global_step)
        self._writer.add_scalar('memory/RAM/local/percent_used',
                                ivy.percent_used_mem_on_dev('cpu', process_specific=True), global_step)
        for ds in self._spec.dev_strs:
            if 'gpu' not in ds:
                continue
            ds_formatted = ds.replace(':', '_').capitalize()
            self._writer.add_scalar('memory/{}/global/percent_used'.format(ds_formatted),
                                    ivy.percent_used_mem_on_dev(ds), global_step)

    def _log_device_utilization(self, global_step):
        if not ivy.exists(self._writer):
            raise Exception('torch must be installed in order to use the file writer for tensorboard logging.')
        self._writer.add_scalar('dev_util/CPU', ivy.dev_util('cpu'), global_step)
        for ds in self._spec.dev_strs:
            if 'gpu' not in ds:
                continue
            ds_formatted = ds.replace(':', '_').capitalize()
            self._writer.add_scalar('dev_util/{}'.format(ds_formatted), ivy.dev_util(ds), global_step)

    # noinspection PyProtectedMember
    def _log_device_tuning(self, global_step):
        if not ivy.exists(self._writer):
            raise Exception('torch must be installed in order to use the file writer for tensorboard logging.')
        if not ivy.exists(self._dev_manager):
            raise Exception('Cannot log device manager tuning if the device manager does not exist.'
                            'Please set either of the params: tune_device_allocation=True or tune_splitting=True')

        # device allocation
        # ToDo: log more useful tuning metrics here
        if self._multi_dev and self._spec.tune_device_allocation:
            self._writer.add_scalar('dev_tuning/device_alloc/tune_count',
                                    self._dev_manager._da_tune_count, global_step)
            self._writer.add_scalar('dev_tuning/device_alloc/unit_tune_count',
                                    self._dev_manager._unit_da_tune_count, global_step)
            self._writer.add_scalar('dev_tuning/device_alloc/step_time',
                                    self._dev_manager._da_step_time, global_step)
            for ds, split in self._dev_manager._dev_strs_da.items():
                self._writer.add_scalar('dev_tuning/device_alloc/split_sizes/{}'.format(ds), split, global_step)

        # per-device splitting
        # ToDo: log more useful tuning metrics here
        if self._spec.tune_splitting:
            self._writer.add_scalar('dev_tuning/splitting/tune_count',
                                    self._dev_manager._ds_tune_count, global_step)
            self._writer.add_scalar('dev_tuning/splitting/step_time',
                                    self._dev_manager._ds_step_time, global_step)
            for ds, split in self._dev_manager._dev_strs_ds.items():
                self._writer.add_scalar('dev_tuning/splitting/split_factors/{}'.format(ds), split, global_step)

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
        first_batch = self._spec.data_loader.get_first_batch()
        if ivy.exists(self._dev_manager):
            self._dev_manager.dim_size = first_batch.shape[0]
        # for on_call builds
        self._compute_cost(self._network, first_batch[0:1], self._spec.dev_strs[0])
        # compile
        if self._spec.compile_graph:
            valid_modes = ['network', 'optimizer', 'all']
            assert self._spec.compile_graph in ['network', 'optimizer', 'all'], 'invalid value for compile_graph, ' \
                                                                                'must be one of {}'.format(valid_modes)
            if self._spec.compile_graph in ['network', 'all']:
                self._compile_network_once_tuned = True
            if self._spec.compile_graph in ['optimizer', 'all']:
                self._compile_optimizer_once_tuned = True
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

    def _raw_execute_with_grads(self, network, dev_str, batch, network_v):
        cost, gradients = ivy.execute_with_gradients(
            lambda v: self._compute_cost(network, batch, dev_str, v=network_v.set_at_key_chains(v)),
            network_v.at_key_chains(self._net_spec.v_keychains, ignore_none=True) if
            self._net_spec.keep_v_keychains else
            network_v.prune_key_chains(self._net_spec.v_keychains, ignore_none=True))
        return cost, gradients

    def _dev_manager_execute_with_grads(self, network, batch):
        # ToDo: assign this function in constructor rather than performing checks on each training step
        dev_manager_exists = ivy.exists(self._dev_manager)
        tuned = not dev_manager_exists or self._dev_manager.tuned
        if self._compile_network_once_tuned and tuned:
            network.compile_on_next_step()
            self._compile_network_once_tuned = False
        if self._compile_optimizer_once_tuned and tuned:
            self._optimizer.compile_on_next_step()
            self._compile_optimizer_once_tuned = False
        if ivy.exists(self._dev_manager):
            if self._multi_dev:
                if not isinstance(batch, ivy.MultiDevContainer):
                    batch = batch.to_multi_dev(self._spec.dev_strs)
                return self._dev_manager.map(distributed={"batch": batch.at_devs()},
                                             to_clone={"network_v": network.v})
            ret = self._raw_execute_with_grads(network, self._spec.dev_strs[0], batch, network.v)
            self._dev_manager.tune_step()
            return ret
        return self._raw_execute_with_grads(network, self._spec.dev_strs[0], batch, network.v)

    def _optimizer_step(self, v, grads):
        # ToDo: consider moving this code to the ivy.Optimizer class
        if 'max_grad_val' in self._spec:
            grads = grads.clip(-self._spec.max_grad_val, self._spec.max_grad_val)
        if 'max_grad_vector_norm' in self._spec:
            ratio = self._spec.max_grad_vector_norm/(grads.vector_norm(global_norm=True) + MIN_DENOMINATOR)
            if ratio < 1:
                grads = grads * ratio
        return self._optimizer.step(v, grads, ignore_missing=self._partial_grad_updates)

    def _train_step_from_batch(self, batch):
        cost, self._gradients = self._dev_manager_execute_with_grads(self._network, batch)
        self._moving_average_loss = (cost + self._global_step * self._moving_average_loss) / (self._global_step + 1)
        return batch, cost, self._optimizer_step(self._network.v, self._gradients)

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
                ivy.try_use_compiled = False
                self._log_scalars()
                ivy.try_use_compiled = True
            if log_viz or vis_mode:
                ivy.try_use_compiled = False
                self._write_image_summaries(self._spec.data_loader, self._network, self._training_batch,
                                            self._global_step)
                ivy.try_use_compiled = True
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
        if ivy.exists(self._dev_manager):
            self._dev_manager.__del__()
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
