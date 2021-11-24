# global
import os
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import ivy
import math
import json
import logging
import collections
import numpy as np
import multiprocessing
from ivy_builder.dataset import Dataset
from ivy.core.container import Container
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.data_loaders.specs.seq_data_loader_spec import SeqDataLoaderSpec


# noinspection PyUnresolvedReferences
class SeqDataLoader(DataLoader):

    def __init__(self, data_loader_spec: SeqDataLoaderSpec):
        super(SeqDataLoader, self).__init__(data_loader_spec)

        # cpus
        if 'num_workers' in data_loader_spec:
            self._total_num_workers = data_loader_spec.num_workers
        else:
            self._total_num_workers = multiprocessing.cpu_count()

        # first frame validity
        if 'first_frame_validity_fn' in data_loader_spec:
            self._first_frame_validity_fn = data_loader_spec.first_frame_validity_fn
        else:
            self._first_frame_validity_fn = None

        # data loader specification
        self._spec = data_loader_spec
        self._container_data_dir = os.path.join(self._spec.dataset_spec.dirs.dataset_dir, 'containers/')
        self._batch_size = self._spec.batch_size
        self._base_cache_size = self._spec.cache_size * self._spec.batch_size * self._spec.window_size
        self._window_size = self._spec.window_size
        start_idx = self._spec.starting_idx
        end_idx = start_idx + self._spec.num_sequences - 1

        # specs before pruning via containers_to_skip
        self._spec.dataset_spec.unpruned_sequence_lengths = self._spec.dataset_spec.sequence_lengths
        self._spec.unpruned_num_sequences = self._spec.num_sequences

        # sequence lengths and windows per sequence
        if 'sequence_lengths' in self._spec.dataset_spec:
            self._fixed_sequence_length = isinstance(self._spec.dataset_spec.sequence_lengths, int)
            if self._fixed_sequence_length:
                self._windows_per_seq = self._spec.dataset_spec.sequence_lengths - (self._window_size - 1)
            else:
                # update sequences lengths
                self._spec.dataset_spec.sequence_lengths =\
                    [self._compute_seq_len(i, sl, self._spec.containers_to_skip)
                     for i, sl in enumerate(self._spec.dataset_spec.sequence_lengths)]
                self._spec.num_sequences =\
                    sum([sl > 0 for sl in self._spec.dataset_spec.sequence_lengths[start_idx:end_idx+1]])
                self._windows_per_seq = ivy.array(self._spec.dataset_spec.sequence_lengths) - (self._window_size - 1)
        else:
            self._fixed_sequence_length = False

        # new end idx following containers_to_skip pruning
        end_idx = start_idx + self._spec.num_sequences - 1

        # compute num workers for each component
        self._compute_num_workers()

        # custom init
        self._custom_init_fn = self._spec.custom_init_fn
        if ivy.exists(self._custom_init_fn):
            self._custom_init_fn(self)

        # dataset
        self._dataset = self._get_dataset(start_idx, end_idx)
        self._iterator = iter(self._dataset)

        # dummy batch
        self._first_batch = None

        # counter
        self._counter = 0

    # Built-Ins #
    # ----------#

    def __del__(self):
        self.close()

    # Helpers #
    # --------#

    @staticmethod
    def _skip_cont(seq_idx, win_idx, conts_to_skip):
        if (seq_idx, None) in conts_to_skip or (None, win_idx) in conts_to_skip or \
                (seq_idx, win_idx) in conts_to_skip:
            return True
        return False

    @staticmethod
    def _compute_seq_len(i, sl, conts_to_skip):
        if (i, None) in conts_to_skip:
            return 0
        return sl - sum([c[0] == i or c[0] is None and c[1] < sl for c in conts_to_skip])

    # Dataset in RAM #
    # ---------------#

    @staticmethod
    def _to_tensor(x, key_chain=''):
        if type(x) == str:
            x = [[list(x.encode())]]
            return ivy.array(x, dtype_str='uint8')
        return ivy.array(x, dtype_str='float32')

    @staticmethod
    def _load_container_filepaths_as_lists(cont_dir, starting_example, ending_example):

        if not os.path.isdir(cont_dir):
            raise Exception('container dir ' + cont_dir + ' does not exist')

        cont_filenames = os.listdir(cont_dir)
        cont_filenames.sort()
        cont_paths = [os.path.join(cont_dir, cont_filename) for cont_filename in cont_filenames]

        grouped_conts = []
        group = []
        cont_num = 0
        for i, cont_path in enumerate(cont_paths):
            seq_id = int(cont_path.split('/')[-1].split('_')[0])
            if seq_id < starting_example:
                continue
            elif seq_id > ending_example:
                break
            trimmed_string = cont_path.split('/')[-1].split('_')[-2]
            current_cont_num = int(trimmed_string) if str.isdigit(trimmed_string) else int(trimmed_string[1:])
            if current_cont_num != cont_num and seq_id != starting_example:
                cont_num = current_cont_num
                grouped_conts.append(group)
                group = []
            group.append(cont_path)
        grouped_conts.append(group)
        return grouped_conts

    def _get_containers_w_filepath_img_entries_as_tensor_slices(self, container_filepaths):
        # noinspection PyUnusedLocal
        all_containers = list()
        logging.info('loading containers into RAM...')
        num_seqs = len(container_filepaths)
        max_seq_len = max(max(len(item) for item in container_filepaths), self._window_size)
        for seq_idx, seq in enumerate(container_filepaths):
            if seq_idx % 10000 == 0:
                logging.info('sequence {} of {}'.format(seq_idx, num_seqs))
            window_containers = list()
            container = None
            seq_len = 0
            for seq_len, filepath in enumerate(seq):
                if filepath == '':
                    seq_len -= 1
                    break
                with open(filepath) as fp:
                    container_dict = json.load(fp)
                container = Container(container_dict).map(self._to_tensor)
                window_containers.append(container)
            window_containers += [container] * (max_seq_len - seq_len - 1)  # padding for shorter sequences
            joined_window_containers = Container.concat(window_containers, 1)
            all_containers.append(joined_window_containers)
        return Container.concat(all_containers, 0)

    # Multiprocessing Sizes #
    # ----------------------#

    def _compute_num_workers(self):

        # init
        num_workers = self._total_num_workers
        self._num_workers = ivy.Container()

        # prefetch
        self._num_workers.prefetch = int(self._spec.with_prefetching) + 1
        num_workers = math.ceil(num_workers/self._num_workers.prefetch)

        # post processed
        self._num_workers.post_processed = 1

        # from numpy
        self._num_workers.from_np = 1

        # batched
        self._num_workers.batched = 1

        # loaded data
        self._num_workers.loaded_data = min(num_workers, self._batch_size)

        # ToDo: add multi-processing support for these lower level datasets

        # shuffled
        self._num_workers.shuffled = 1

        # unbatch
        self._num_workers.unbatched = 1

        # windowed
        self._num_workers.windowed = 1

        # valid first frames
        self._num_workers.valid_first_frames = 1

        # keychain pruned
        self._num_workers.keychain_pruned = 1

        # parsed json
        self._num_workers.parsed_json = 1

        # loaded json
        self._num_workers.loaded_json = 1

    # Dynamic Windowing #
    # ------------------#

    def _update_seq_info_for_window(self, seq_info):
        if not ivy.exists(seq_info):
            return
        seq_idx = int(seq_info.seq_idx[0])
        seq_len = int(seq_info.length[0])
        new_len = self._compute_seq_len(seq_idx, seq_len, self._spec.containers_to_skip)
        seq_info = seq_info.copy()
        seq_info.length = ivy.ones_like(seq_info.length) * new_len
        return seq_info

    def _group_tensor_into_windowed_tensor_simple(self, x, seq_info):
        seq_info = self._update_seq_info_for_window(seq_info)
        if self._fixed_sequence_length:
            return ivy.reshape(ivy.gather_nd(x, ivy.array(self._gather_idxs)),
                               (self._windows_per_seq, self._window_size) + x.shape[1:])
        else:
            num_windows_in_seq = int(ivy.to_numpy(ivy.maximum(seq_info.length[0] - self._window_size + 1, 1)))
            window_idxs_in_seq = ivy.arange(num_windows_in_seq, 0, 1)
            gather_idxs = ivy.tile(ivy.reshape(ivy.arange(self._window_size, 0, 1), (1, self._window_size)),
                                   (num_windows_in_seq, 1)) + ivy.expand_dims(window_idxs_in_seq, -1)
            gather_idxs_flat = ivy.reshape(gather_idxs, (self._window_size * num_windows_in_seq, 1))
            return ivy.reshape(ivy.gather_nd(x, gather_idxs_flat),
                               (num_windows_in_seq, self._window_size) + x.shape[1:])

    def _group_tensor_into_windowed_tensor(self, x, valid_first_frame):
        if self._window_size == 1:
            valid_first_frame_pruned = ivy.cast(valid_first_frame[:, 0], 'bool')
        else:
            valid_first_frame_pruned = ivy.cast(valid_first_frame[:1-self._window_size, 0], 'bool')
        if ivy.reduce_sum(ivy.cast(valid_first_frame_pruned, 'int32'))[0] == 0:
            valid_first_frame_pruned =\
                ivy.cast(ivy.one_hot(0, self._sequence_lengths[0] - self._window_size + 1), 'bool')
        window_idxs_single = ivy.indices_where(valid_first_frame_pruned)

        gather_idxs_list = list()
        for w_idx in window_idxs_single:
            gather_idxs_list.append(ivy.expand_dims(ivy.arange(w_idx[0] + self._window_size, w_idx[0], 1), 0))
        gather_idxs = ivy.concatenate(gather_idxs_list, 0)
        gather_idxs = ivy.reshape(gather_idxs, (-1, 1))
        num_valid_windows_for_seq = ivy.shape(window_idxs_single)[0:1]
        return ivy.reshape(ivy.gather_nd(x, gather_idxs),
                           ivy.concatenate((num_valid_windows_for_seq,
                                            ivy.array([self._window_size]), ivy.shape(x)[1:]), 0))

    def _group_container_into_windowed_container(self, container):
        if self._first_frame_validity_fn is not None:
            return container.map(lambda x, _: self._group_tensor_into_windowed_tensor(x, container.valid_first_frame))
        else:
            if 'seq_info' in container:
                seq_info = container.seq_info
            else:
                seq_info = None
            return container.map(lambda x, _: self._group_tensor_into_windowed_tensor_simple(x, seq_info))

    # Dynamic File Reading #
    # ---------------------#

    # json to container

    @staticmethod
    def _load_json_files(containers):
        read_files = list()
        for j_fpath in containers.fpaths:
            if j_fpath != '':
                with open(j_fpath, 'r') as file:
                    read_str = file.read()
            else:
                read_str = ''
            read_files.append(read_str)
        return ivy.Container({'json_str': read_files})

    def _parse_json_strings(self, containers):
        json_strings_stack = containers.json_str
        highest_idx_entry = len([item for item in containers.json_str if item != '']) - 1
        json_container_stack = [Container(json.loads(json_str)).map(self._to_tensor)[0]
                                if json_str != '' else
                                Container(json.loads(json_strings_stack[highest_idx_entry])).map(
                                    self._to_tensor)[0] for json_str in json_strings_stack]
        return Container.concat(json_container_stack, 0)

    # container pruning

    def _prune_unused_key_chains(self, container):
        container = container.prune_key_chains(self._spec.unused_key_chains)
        return container

    # arrays

    def _array_fn(self, filepaths_in_window):
        conts = list()
        for filepath in filepaths_in_window:
            str_path = bytearray(ivy.to_numpy(filepath).tolist()).decode()
            full_path = os.path.abspath(os.path.join(self._container_data_dir, str_path))
            if self._spec.array_mode == 'hdf5':
                cont = ivy.Container.from_disk_as_hdf5(full_path + '.hdf5')
            elif self._spec.array_mode == 'pickled':
                cont = ivy.Container.from_disk_as_pickled(full_path + '.pickled')
            else:
                raise Exception('array_mode must be one of [ hdf5 | pickled ],'
                                'but found {}'.format(self._spec.array_mode))
            conts.append(cont)
        return ivy.Container.concat(conts, 0)

    # images

    def _uint8_img_fn(self, filepaths_in_window):
        imgs = list()
        for filepath in filepaths_in_window:
            str_path = bytearray(ivy.to_numpy(filepath).tolist()).decode()
            full_path = os.path.abspath(os.path.join(self._container_data_dir, str_path))
            if not ivy.exists(cv2):
                raise Exception('in order to use _uint8_img_fn, opencv for python must be installed.'
                                'To install opencv, run pip install opencv-python.')
            img_rgb = cv2.imread(full_path, -1)
            if len(img_rgb.shape) == 2:
                if not self._spec.load_gray_as_rgb:
                    raise Exception('Found an image with shape {}, but load_gray_as_rgb is set to False.'
                                    'Set this to True in order to tile grayscale images to RGB.'.format(img_rgb.shape))
                img_rgb = np.tile(np.expand_dims(img_rgb, -1), (1, 1, 3))
            img = ivy.array(np.expand_dims(img_rgb.astype(np.float32), 0))/255
            imgs.append(img)
        return ivy.concatenate(imgs, 0)

    def _float_img_fn(self, filepaths_in_window):
        imgs = list()
        for filepath in filepaths_in_window:
            str_path = bytearray(ivy.to_numpy(filepath).tolist()).decode()
            full_path = os.path.abspath(os.path.join(self._container_data_dir, str_path))
            if not ivy.exists(cv2):
                raise Exception('in order to use _float_img_fn, opencv for python must be installed.'
                                'To install opencv, run pip install opencv-python.')
            img_rgba = cv2.imread(full_path, -1)
            img = ivy.array(np.frombuffer(img_rgba.tobytes(), np.float32).reshape((1,) + img_rgba.shape[:-1]))
            imgs.append(img)
        return ivy.concatenate(imgs, 0)

    def _custom_img_fn(self, filepaths_in_window, fn):
        imgs = list()
        for filepath in filepaths_in_window:
            str_path = bytearray(ivy.to_numpy(filepath).tolist()).decode()
            full_path = os.path.abspath(os.path.join(self._container_data_dir, str_path))
            if not ivy.exists(cv2):
                raise Exception('in order to use _custom_img_fn, opencv for python must be installed.'
                                'To install opencv, run pip install opencv-python.')
            img_raw = cv2.imread(full_path, -1)
            img = fn(img_raw)
            imgs.append(img)
        img0 = imgs[0]
        if isinstance(img0, ivy.Container):
            return ivy.Container.concat(imgs, 0)
        elif ivy.is_array(img0):
            return ivy.concatenate(imgs, 0)
        else:
            raise Exception('custom image functions should either return an array or an ivy.Container instance,'
                            'but found {} or type {}'.format(img0, type(img0)))

    def _str_fn(self, x, key_chain=''):
        for array_str in self._spec.array_strs:
            if array_str in key_chain:
                return self._array_fn(x)
        for float_str in self._spec.float_strs:
            if float_str in key_chain:
                return self._float_img_fn(x)
        for uint8_str in self._spec.uint8_strs:
            if uint8_str in key_chain:
                return self._uint8_img_fn(x)
        for i, custom_img_strs in enumerate(self._spec.custom_img_strs):
            for custom_img_str in custom_img_strs:
                if custom_img_str in key_chain:
                    return self._custom_img_fn(x, self._spec.custom_img_fns[i])
        for i, custom_strs in enumerate(self._spec.custom_strs):
            for custom_str in custom_strs:
                if custom_str in key_chain:
                    return self._spec.custom_fns[i](x, self._container_data_dir)
        return x

    def _load_data_from_filepath_tensors(self, container):
        return container.map(self._str_fn)

    # Dataset Creation #
    # -----------------#

    def _get_dataset(self, starting_example, ending_example):

        class ContainerIdxMap:

            def __init__(self, sizes, fpath_template=None, seq_idxs=None, start=None, end=None, max_seq_len=None,
                         conts_to_skip=None, pruned_sizes=None):
                if isinstance(sizes, (tuple, list)):
                    pruned_sizes = ivy.default(
                        pruned_sizes, [SeqDataLoader._compute_seq_len(i, sl, conts_to_skip)
                                       for i, sl in enumerate(sizes)])
                    num_empty = sum([ps == 0 for ps in pruned_sizes])
                    self._raw_sizes = dict(zip(range(start, end + 1 + num_empty),
                                               sizes[start:end + 1 + num_empty]))
                    self._pruned_sizes = dict(zip(range(start, end + 1 + num_empty),
                                                  pruned_sizes[start:end + 1 + num_empty]))
                elif isinstance(sizes, (int, dict)):
                    self._raw_sizes = sizes
                    self._pruned_sizes = ivy.default(pruned_sizes, sizes)
                    if isinstance(self._pruned_sizes, int):
                        pruned_dict = dict()
                        for seq_idx, win_idx in conts_to_skip:
                            if seq_idx not in pruned_dict:
                                pruned_dict[seq_idx] = list()
                            pruned_dict[seq_idx].append(win_idx)
                        pruned_dict = {k: len(set(v)) for k, v in pruned_dict.items()}
                        pruned_sizes_dict = {k: self._pruned_sizes - num_pruned
                                             for k, num_pruned in pruned_dict.items()}
                        num_empty = sum([size == 0 for size in pruned_sizes_dict.values()])
                        pruned_sizes = collections.defaultdict(
                            lambda: self._pruned_sizes, pruned_sizes_dict)
                    else:
                        num_empty = sum([ps == 0 for ps in self._pruned_sizes])
                else:
                    raise Exception('Invalid type for sizes, expected one of int, dict, tuple or list,'
                                    'but found {} or type {}'.format(sizes, type(sizes)))
                self._constant_size = isinstance(self._raw_sizes, int)
                if max_seq_len:
                    self._max_seq_len = max_seq_len
                else:
                    self._max_seq_len = self._pruned_sizes if self._constant_size else max(self._pruned_sizes.values())
                self._fpath_template = fpath_template
                self._conts_to_skip = conts_to_skip
                if seq_idxs:
                    self._seq_idxs = seq_idxs
                else:
                    vals = [v for i, v in enumerate(range(start, end + 1 + num_empty)) if pruned_sizes[i] > 0]
                    keys = range(0, min(end - start + 1 + num_empty, len(vals)))
                    self._seq_idxs = dict(zip(keys, vals))

            def __getitem__(self, slice_obj):
                if isinstance(slice_obj, slice):
                    seq_idxs = collections.OrderedDict(
                        [(i, self._seq_idxs[idx]) for i, idx in
                         enumerate(range(slice_obj.start, slice_obj.stop, ivy.default(slice_obj.step, 1)))])
                elif isinstance(slice_obj, int):
                    seq_idxs = collections.OrderedDict({0: self._seq_idxs[slice_obj]})
                else:
                    raise Exception('Invalid type for slice_obj, expected either slice or int,'
                                    'but found {} of type {}'.format(slice_obj, type(slice_obj)))
                if self._constant_size:
                    sizes = self._raw_sizes
                else:
                    sizes = collections.OrderedDict({seq_idx: self._raw_sizes[seq_idx]
                                                     for seq_idx in seq_idxs.values()})
                return ContainerIdxMap(sizes, self._fpath_template, seq_idxs, max_seq_len=self._max_seq_len,
                                       conts_to_skip=self._conts_to_skip, pruned_sizes=self._pruned_sizes)

            def __len__(self):
                return len(self._seq_idxs)

            def shuffle(self):
                mapped_idxs = list(self._seq_idxs.values())
                np.random.shuffle(mapped_idxs)
                self._seq_idxs = collections.OrderedDict(zip(self._seq_idxs.keys(), mapped_idxs))

            def to_idxs(self):
                seq_idxs = self._seq_idxs.values()
                sizes = [self._raw_sizes if self._constant_size else self._raw_sizes[seq_idx] for seq_idx in seq_idxs]
                rets = [[(seq_idx, win_idx) for win_idx in
                         range(size) if not SeqDataLoader._skip_cont(seq_idx, win_idx, self._conts_to_skip)]
                        for seq_idx, size in zip(seq_idxs, sizes)]
                return [r + [(None, None)] * (self._max_seq_len - len(r)) for r in rets if list(set(r)) != [None]]

            def to_filepaths(self):
                if not ivy.exists(self._fpath_template):
                    raise Exception('to_filepaths method is not valid if fpath_template has not been specified'
                                    'in the constructor.')
                seq_idxs = self._seq_idxs.values()
                sizes = [self._raw_sizes if self._constant_size else self._raw_sizes[seq_idx] for seq_idx in seq_idxs]
                rets = [[self._fpath_template % (seq_idx, win_idx) for win_idx in
                         range(size) if not SeqDataLoader._skip_cont(seq_idx, win_idx, self._conts_to_skip)]
                        for seq_idx, size in zip(seq_idxs, sizes)]
                return [r + [''] * (self._max_seq_len - len(r)) for r in rets if ''.join(r) != '']

            @property
            def sizes(self):
                return self._pruned_sizes

        # container filepaths
        if self._spec.container_load_mode in ['preload', 'dynamic']:
            fpath_template = os.path.join(self._container_data_dir, self._spec.dataset_spec.cont_fname_template)
        else:
            fpath_template = None
        container_idx_map = ContainerIdxMap(
            self._spec.dataset_spec.unpruned_sequence_lengths, fpath_template, start=starting_example,
            end=ending_example, conts_to_skip=self._spec.containers_to_skip)

        if self._spec.num_sequences != -1:
            container_idx_map = container_idx_map[0:self._spec.num_sequences]

        # shuffle sequences
        if self._spec.preshuffle_data:
            container_idx_map.shuffle()

        # extract sequence lengths
        if self._fixed_sequence_length:
            self._sequence_lengths =\
                collections.OrderedDict(zip(range(len(container_idx_map)),
                                            [self._spec.dataset_spec.sequence_lengths] * len(container_idx_map)))
            self._windows_per_seq = self._sequence_lengths[0] - self._window_size + 1
            # windowing values
            window_idxs_per_seq = ivy.reshape(ivy.arange(self._windows_per_seq, 0, 1), (self._windows_per_seq, 1))
            gather_idxs_list = list()
            for x in window_idxs_per_seq:
                gather_idxs_list.append(ivy.expand_dims(ivy.arange(x[0] + self._window_size, x[0], 1), 0))
            gather_idxs = ivy.concatenate(gather_idxs_list, 0)
            self._gather_idxs = \
                ivy.to_numpy(ivy.reshape(gather_idxs, (self._windows_per_seq * self._window_size, 1))).tolist()
        else:
            self._sequence_lengths = container_idx_map.sizes

        # maybe pre-load containers
        if self._spec.container_load_mode == 'preload':
            # load containers with vector data and image filepath entries
            container_slices = self._get_containers_w_filepath_img_entries_as_tensor_slices(
                container_idx_map.to_filepaths())
            if self._first_frame_validity_fn is not None:
                container_slices =\
                    self._first_frame_validity_fn(container_slices, [ending_example - starting_example + 1])

            # prune unwanted chains of keys
            if 'unused_key_chains' in self._spec:
                container_slices = self._prune_unused_key_chains(container_slices)

            dataset = Dataset(
                ivy.Container.list_stack([c[0] for c in container_slices.unstack(0, container_slices.shape[0])], 0),
                'base',
                container_slices.shape[0],
                numpy_loading=True,
                cache_size=self._base_cache_size,
                queue_timeout=self._spec.queue_timeout)
        else:
            if self._spec.container_load_mode == 'dynamic':
                # load containers with filepath entries
                dataset = Dataset(ivy.Container({'fpaths': container_idx_map}),
                                  'base',
                                  len(container_idx_map),
                                  trans_fn=lambda cont: cont.map(lambda x_, kc: x_.to_filepaths()),
                                  elementwise_query_fn=False,
                                  numpy_loading=True,
                                  cache_size=self._base_cache_size,
                                  queue_timeout=self._spec.queue_timeout)
                dataset = dataset.map('loaded_json',
                                      self._load_json_files,
                                      self._num_workers.loaded_json)
                dataset = dataset.map('parsed_json',
                                      self._parse_json_strings,
                                      self._num_workers.parsed_json)
            else:
                dataset = Dataset(ivy.Container({'idx_map': container_idx_map}),
                                  'base',
                                  len(container_idx_map),
                                  trans_fn=lambda cont: self._spec.custom_container_load_fn(self, cont),
                                  elementwise_query_fn=False,
                                  numpy_loading=True,
                                  cache_size=self._base_cache_size,
                                  queue_timeout=self._spec.queue_timeout)
            if 'unused_key_chains' in self._spec:
                dataset = dataset.map('keychain_pruned',
                                      self._prune_unused_key_chains,
                                      self._num_workers.keychain_pruned)
            if self._first_frame_validity_fn is not None:
                dataset = dataset.map('valid_first_frames',
                                      lambda x_: self._first_frame_validity_fn(x_, None),
                                      self._num_workers.valid_first_frames)
        if not (self._spec.dataset_spec.sequence_lengths == 1 and self._window_size == 1):
            # ToDo: add other conditionals which make the loading more efficient if only one of the
            #  above two conditions is True
            dataset = dataset.map('windowed',
                                  self._group_container_into_windowed_container,
                                  self._num_workers.windowed)
            dataset = dataset.unbatch('unbatched',
                                      self._num_workers.unbatched,
                                      batch_sizes=[max(seq_len, self._window_size) - self._window_size + 1
                                                   for seq_len in self._sequence_lengths.values() if seq_len > 0])
        if self._spec.shuffle_buffer_size > 0:
            dataset = dataset.shuffle('shuffled',
                                      self._spec.shuffle_buffer_size,
                                      self._num_workers.shuffled)
        dataset = dataset.map('loaded_data',
                              self._load_data_from_filepath_tensors,
                              self._num_workers.loaded_data)
        dataset = dataset.batch('batched',
                                self._batch_size,
                                self._num_workers.batched)
        dataset = dataset.map('from_np',
                              lambda cont: cont.map(lambda x_, kc: ivy.array(x_)),
                              self._num_workers.from_np,
                              numpy_loading=False)
        if ivy.exists(self._spec.post_proc_fn):
            dataset = dataset.map('post_processed',
                                  self._spec.post_proc_fn,
                                  self._num_workers.post_processed)
        if self._spec.with_prefetching:
            dataset = dataset.prefetch('prefetch')
        # ToDo: find way to make pre-fetching to GPU actually pre-fetch, ideally using multi-processing.
        #  For example, swapping prefetch and to_gpu ops around would work if to_gpu could accept self._num_workers.
        if self._spec.prefetch_to_devs:
            if isinstance(self._spec.prefetch_to_devs, str):
                dataset = dataset.to_dev('to_dev', self._spec.prefetch_to_devs)
            elif len(self._spec.prefetch_to_devs) == 1:
                dataset = dataset.to_dev('to_dev', self._spec.prefetch_to_devs[0])
            else:
                dataset = dataset.to_devs('to_devs', self._spec.prefetch_to_devs)
        return dataset

    # Public Methods #
    # ---------------#

    def get_next_batch(self, dataset_key=None):
        return next(self._iterator)

    def get_first_batch(self, dataset_key=None):
        if self._first_batch is None:
            self._first_batch = self._dataset[0]
        return self._first_batch

    def cycle_for_debugging(self, offset=0):
        self._dataset.cycle_for_debugging(offset)

    def close(self):
        self._dataset.close()
