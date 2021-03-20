# global
import os
import json
import random
import logging
import multiprocessing
import tensorflow as tf
import tensorflow_io as tfio
from ivy.core.container import Container
from ivy_builder.abstract.data_loader import DataLoader
from ivy_builder.specs import DataLoaderSpec


# noinspection PyUnresolvedReferences
class TFDataLoader(DataLoader):

    def __init__(self, data_loader_spec: DataLoaderSpec):
        super(TFDataLoader, self).__init__(data_loader_spec)

        # prevents QT conflicts with other applications, such as PyRep
        import cv2
        self._cv2 = cv2

        # cpus
        if 'num_workers' in data_loader_spec:
            self._num_workers = data_loader_spec.num_workers
        else:
            self._num_workers = multiprocessing.cpu_count()

        # first frame validity
        if 'first_frame_validity_fn' in data_loader_spec:
            self._first_frame_validity_fn = data_loader_spec.first_frame_validity_fn
        else:
            self._first_frame_validity_fn = None

        # data loader specification
        self._spec = data_loader_spec
        self._container_data_dir = os.path.join(self._spec.dataset_spec.dirs.dataset_dir, 'containers')
        self._container_data_dir_tensor = tf.constant(self._container_data_dir + '/', tf.string)

        # TF variables
        self._window_size = self._spec.window_size
        if 'sequence_lengths' in self._spec:
            self._fixed_sequence_length = isinstance(self._spec.dataset_spec.sequence_lengths, int)
            self._windows_per_seq = tf.constant(self._spec.dataset_spec.sequence_lengths) - (self._window_size - 1)
        else:
            self._fixed_sequence_length = False
        self._batch_size = self._spec.batch_size

        # parallelism
        self._parallel_window_iterations = min(self._window_size, self._num_workers)

        # train and validation idxs
        start_idx_train = 0
        end_idx_train = self._spec.num_training_sequences - 1
        start_idx_valid = self._spec.num_training_sequences
        end_idx_valid = self._spec.num_sequences_to_use - 1

        # train dataset
        train_dataset = self._get_dataset(start_idx_train, end_idx_train)
        self._training_iterator = iter(train_dataset)

        # validation
        if self._spec.num_training_sequences < self._spec.num_sequences_to_use:
            validation_dataset = self._get_dataset(start_idx_valid, end_idx_valid)
            self._validation_iterator = iter(validation_dataset)
        else:
            self._validation_iterator = None

        # dummy batch
        self._dummy_batch = None

    # Dataset in RAM #
    # ---------------#

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
        def _to_tensor(x, key_chain=''):
            if type(x) == str:
                x = [[x]]
            return tf.constant(x)

        all_containers = list()
        logging.info('loading containers into RAM...')
        num_seqs = len(container_filepaths)
        max_seq_len = max(max(len(item) for item in container_filepaths), self._window_size)
        for seq_idx, seq in enumerate(container_filepaths):
            if seq_idx % 10000 == 0:
                logging.info('sequence {} of {}'.format(seq_idx, num_seqs))
            window_containers = list()
            tf_container = None
            seq_len = 0
            for seq_len, filepath in enumerate(seq):
                if filepath == '':
                    seq_len -= 1
                    break
                with open(filepath) as fp:
                    container_dict = json.load(fp)
                tf_container = Container(container_dict).map(_to_tensor)
                window_containers.append(tf_container)
            window_containers += [tf_container] * (max_seq_len - seq_len - 1)  # padding for shorter sequences
            joined_window_containers = Container.concat(window_containers, 1, tf)
            all_containers.append(joined_window_containers)
        return Container.concat(all_containers, 0, tf)

    # Dynamic Windowing #
    # ------------------#

    def _group_tensor_into_windowed_tensor_simple(self, x, seq_info):
        if self._fixed_sequence_length:
            return tf.reshape(tf.gather_nd(x, self._gather_idxs),
                              [self._windows_per_seq, self._window_size] + x.shape[1:])
        else:
            num_windows_in_seq = tf.maximum(seq_info.length[0] - self._window_size + 1, 1)
            window_idxs_in_seq = tf.range(0, num_windows_in_seq, 1)
            gather_idxs = tf.tile(tf.reshape(tf.range(0, self._window_size, 1), (1, self._window_size)),
                                  (num_windows_in_seq, 1)) + tf.expand_dims(window_idxs_in_seq, -1)
            gather_idxs_flat = tf.reshape(gather_idxs, (self._window_size * num_windows_in_seq, 1))
            return tf.reshape(tf.gather_nd(x, gather_idxs_flat),
                              tf.concat((tf.expand_dims(num_windows_in_seq, 0),
                                        tf.constant([self._window_size]), tf.shape(x)[1:]), 0))

    def _group_tensor_into_windowed_tensor(self, x, valid_first_frame):
        if self._window_size == 1:
            valid_first_frame_pruned = tf.cast(valid_first_frame[:, 0], tf.bool)
        else:
            valid_first_frame_pruned = tf.cast(valid_first_frame[:1-self._window_size, 0], tf.bool)
        if tf.reduce_sum(tf.cast(valid_first_frame_pruned, tf.int32)) == 0:
            valid_first_frame_pruned = tf.cast(tf.one_hot(0, self._sequence_lengths - self._window_size + 1), tf.bool)
        window_idxs_single = tf.where(valid_first_frame_pruned)
        gather_idxs = tf.reshape(tf.map_fn(lambda x_: tf.range(x_[0], x_[0] + self._window_size, 1),
                                           window_idxs_single), (-1, 1))
        num_valid_windows_for_seq = tf.shape(window_idxs_single)[0:1]
        return tf.reshape(tf.gather_nd(x, gather_idxs),
                          tf.concat((num_valid_windows_for_seq,
                                     tf.constant([self._window_size]),
                                     tf.shape(x)[1:]), 0))

    def _group_container_into_windowed_container(self, container):
        if self._first_frame_validity_fn is not None:
            return container.map(lambda x, _: self._group_tensor_into_windowed_tensor(x, container.valid_first_frame))
        else:
            if 'seq_info' in container:
                seq_info = container.seq_info
            else:
                seq_info = None
            return container.map(lambda x, _:
                                 self._group_tensor_into_windowed_tensor_simple(x, seq_info))

    # Dynamic File Reading #
    # ---------------------#

    # json to container

    def _load_json_files(self, json_filepaths):
        return tf.map_fn(
            lambda json_path: tf.io.read_file(json_path) if json_path != '' else '', json_filepaths,
            parallel_iterations=self._parallel_window_iterations)

    def _parse_json_strings(self, json_strings):
        json_strings_stack = tf.unstack(json_strings)
        highest_idx_entry = tf.reduce_sum(tf.cast(json_strings != '', tf.int32)) - 1
        json_container_stack = [Container(tfio.experimental.serialization.decode_json(
            json_str, self._container_tensor_spec)).slice(0) if json_str != '' else
                                Container(tfio.experimental.serialization.decode_json(
                                    json_strings[highest_idx_entry], self._container_tensor_spec)).slice(0)
                                for json_str in json_strings_stack]
        return Container.concat(json_container_stack, 0, tf)

    # container pruning

    def _prune_unused_key_chains(self, container):
        for unused_key_chain in self._spec.unused_key_chains:
            container = container.prune_key_chain(unused_key_chain)
        return container

    # images

    def _uint8_fn(self, filepaths_in_window):
        return tf.cast(tf.map_fn(
            lambda img_path: tf.image.decode_image(tf.io.read_file(
                tf.strings.join([self._container_data_dir_tensor, img_path]))), filepaths_in_window, dtype=tf.uint8,
            parallel_iterations=self._parallel_window_iterations), tf.float32) / 255

    def _depth_fn(self, filepaths_in_window):
        return tf.bitcast(tf.map_fn(
            lambda depth_img_path: tf.image.decode_image(tf.io.read_file(tf.strings.join(
                [self._container_data_dir_tensor, depth_img_path])), channels=4), filepaths_in_window, dtype=tf.uint8,
            parallel_iterations=self._parallel_window_iterations), tf.float32)

    def _str_fn(self, x, key_chain=''):
        if 'image' in key_chain:
            if 'depth' in key_chain:
                return self._depth_fn(x)
            else:
                return self._uint8_fn(x)
        return x

    def _load_images_from_filepath_tensors(self, container):
        return container.map(self._str_fn)

    # Dataset Creation #
    # -----------------#

    def _get_dataset(self, starting_example, ending_example):

        # container filepaths
        container_filepaths = self._load_container_filepaths_as_lists(self._container_data_dir, starting_example,
                                                                      ending_example)
        max_seq_len = max(max([len(item) for item in container_filepaths]), self._window_size)
        if self._spec.num_sequences_to_use != -1:
            container_filepaths = container_filepaths[0:self._spec.num_sequences_to_use]
        if self._fixed_sequence_length:
            self._sequence_lengths = len(container_filepaths[0])
            self._windows_per_seq = self._sequence_lengths - self._window_size + 1
            # windowing values
            window_idxs_per_seq = tf.reshape(tf.range(0, self._windows_per_seq, 1), (self._windows_per_seq, 1))
            self._gather_idxs = \
                tf.reshape(tf.map_fn(lambda x: tf.range(x[0], x[0] + self._window_size, 1), window_idxs_per_seq),
                           (self._windows_per_seq * self._window_size, 1)).numpy().tolist()
        else:
            self._sequence_lengths = [len(item) for item in container_filepaths]

        # identify which directories are for rgb loading and which are for rgba->float loading
        self._img_channel_dims = dict()
        with open(container_filepaths[0][0]) as fp:
            first_container_dict = json.load(fp)
        first_container = Container(first_container_dict)

        def _to_tensor_spec(value, key_chain=''):
            value_as_tensor = tf.constant(value)
            return tf.TensorSpec(value_as_tensor.shape, value_as_tensor.dtype)

        self._container_tensor_spec = first_container.map(_to_tensor_spec)
        for key, val in first_container.to_iterator():
            if type(val) == str:
                full_filepath = os.path.abspath(os.path.join(self._container_data_dir, val))
                img = self._cv2.imread(full_filepath, -1)
                if img is not None:
                    self._img_channel_dims['/'.join(val.split('/')[:-1])] = img.shape[-1]

        # padding to make rectangular
        container_filepaths = [item + ['']*(max_seq_len - len(item)) for item in container_filepaths]
        random.shuffle(container_filepaths)

        if self._spec.preload_containers:
            # load containers with vector data and image filepath entries
            container_slices = self._get_containers_w_filepath_img_entries_as_tensor_slices(container_filepaths)
            if self._first_frame_validity_fn is not None:
                container_slices =\
                    self._first_frame_validity_fn(container_slices, [ending_example - starting_example + 1])

            # prune unwanted chains of keys
            if 'unused_key_chains' in self._spec:
                container_slices = self._prune_unused_key_chains(container_slices)

            dataset = tf.data.Dataset.from_tensor_slices(container_slices)
        else:
            # load containers with filepath entries
            dataset = tf.data.Dataset.from_tensor_slices(container_filepaths)
            dataset = dataset.map(map_func=self._load_json_files, num_parallel_calls=self._num_workers)
            dataset = dataset.map(map_func=self._parse_json_strings, num_parallel_calls=self._num_workers)
            if 'unused_key_chains' in self._spec:
                dataset = dataset.map(map_func=self._prune_unused_key_chains, num_parallel_calls=self._num_workers)
            if self._first_frame_validity_fn is not None:
                dataset = dataset.map(lambda x: self._first_frame_validity_fn(x, None))
        dataset = dataset.map(map_func=lambda x: self._group_container_into_windowed_container(x),
                              num_parallel_calls=self._num_workers)
        dataset = dataset.unbatch()
        if self._spec.shuffle_buffer_size > 0:
            dataset = dataset.shuffle((max_seq_len - self._window_size + 1) * self._spec.shuffle_buffer_size)
        dataset = dataset.map(map_func=self._load_images_from_filepath_tensors,
                              num_parallel_calls=self._num_workers)
        dataset = dataset.batch(self._batch_size, True)
        if self._spec.post_proc_fn is not None:
            dataset = dataset.map(map_func=self._spec.post_proc_fn, num_parallel_calls=self._num_workers)
        dataset = dataset.prefetch(2)
        if self._spec.prefetch_to_gpu:
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0', 1))
        if not ('single_pass' in self._spec and self._spec.single_pass):
            dataset = dataset.repeat()
        return dataset

    # Public Methods #
    # ---------------#

    def get_next_batch(self, dataset_key):
        if dataset_key == 'training':
            return next(self._training_iterator)
        elif dataset_key == 'validation':
            return next(self._validation_iterator)
        else:
            raise Exception('invalid key')

    def get_next_training_batch(self):
        return self.get_next_batch('training')

    def get_next_validation_batch(self):
        return self.get_next_batch('validation')

    def get_dummy_batch(self):
        if self._dummy_batch is None:
            self._dummy_batch = self.get_next_training_batch()
        return self._dummy_batch.to_random(tf)
