# global
import ivy
import math
import queue
import numbers
import ivy.numpy
import numpy as np
import multiprocessing


# noinspection PyMissingConstructor
class Cache:
    
    def __init__(self, max_size):
        self._max_size = max_size
        self._used_keys = list()
        self._dict = dict()
        
    def __setitem__(self, key, value):
        if key in self:
            self._used_keys.remove(key)
            self._used_keys.append(key)
            return
        self._used_keys.append(key)
        if len(self._used_keys) > self._max_size:
            key_to_del = self._used_keys.pop(0)
            del self._dict[key_to_del]
        self._dict[key] = value
        
    def __getitem__(self, item):
        return self._dict[item]

    def __contains__(self, key):
        return key in self._dict


class Dataset:

    def __init__(self, base_dataset, name, size, base_slice_fn=None, trans_fn=None, slice_fn=None,
                 elementwise_query_fn=True, with_caching=True, cache_size=1, num_processes=1, numpy_loading=False):
        self._copy_args = dict(base_dataset=base_dataset, name=name, size=size, base_slice_fn=base_slice_fn,
                               trans_fn=trans_fn, slice_fn=slice_fn, elementwise_query_fn=elementwise_query_fn,
                               with_caching=with_caching, cache_size=cache_size, num_processes=1)
        self._name = name
        self._size = size
        if base_slice_fn is None:
            self._slice_base_dataset = self._default_base_slice_fn
        else:
            self._slice_base_dataset = base_slice_fn
        self._trans_fn = trans_fn
        if slice_fn is None:
            self._slice_dataset = self._default_slice_fn
        else:
            self._slice_dataset = slice_fn
        self._elementwise_query_fn = elementwise_query_fn
        self._with_caching = with_caching
        self._cache_size = cache_size
        self._cache = Cache(cache_size)
        self._num_processes = multiprocessing.cpu_count() if num_processes is None else num_processes
        self._numpy_loading = numpy_loading
        if numpy_loading and isinstance(base_dataset, ivy.Container):

            def to_numpy(x, kc):
                if ivy.is_array(x):
                    return ivy.to_numpy(x)
                elif isinstance(x, list):
                    return [ivy.to_numpy(v) if ivy.is_array(v) else v for v in x]
                else:
                    return x

            base_dataset = base_dataset.map(to_numpy)
        self._base_dataset = base_dataset
        if self._num_processes > 1:
            self._workers = list()
            self._slice_queues = list()
            self._output_queues = list()
            for i in range(self._num_processes):
                dataset_copy = self._deep_copy()
                index_queue = multiprocessing.Queue()
                output_queue = multiprocessing.Queue()
                worker = multiprocessing.Process(
                    target=self._worker_fn, args=(index_queue, output_queue, dataset_copy, numpy_loading))
                worker.daemon = True
                worker.start()
                self._slice_queues.append(index_queue)
                self._output_queues.append(output_queue)
                self._workers.append(worker)

    # Private #
    # --------#

    def _deep_copy(self):
        base_dataset = self._copy_args['base_dataset']
        if isinstance(base_dataset, ivy.Container):
            return Dataset(**self._copy_args)
        # noinspection PyProtectedMember
        base_dataset = base_dataset._deep_copy()
        copy_args = dict(**self._copy_args)
        copy_args['base_dataset'] = base_dataset
        return Dataset(**copy_args)

    @staticmethod
    def _worker_fn(index_queue, output_queue, dataset, numpy_loading):
        while True:
            try:
                slice_obj = index_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            if slice_obj is None:
                break
            if numpy_loading:
                ivy.set_framework('numpy')
            # noinspection PyProtectedMember
            item = dataset._get_item(slice_obj)
            if numpy_loading:
                ivy.unset_framework()
            output_queue.put(item)

    @staticmethod
    def _is_int(val):
        return abs(round(val) - val) < 1e-6

    @staticmethod
    def _ensure_number_is_int(val):
        val_rounded = round(val)
        if abs(val_rounded - val) > 1e-6:
            raise Exception('Trying to slice ivy Container with non-integer slice {}'.format(val))
        return int(val_rounded)

    @staticmethod
    def _slice_dataset(slice_obj, dataset):
        if isinstance(dataset, ivy.Container):
            if isinstance(slice_obj, numbers.Number):
                slice_obj = Dataset._ensure_number_is_int(slice_obj)
            else:
                so_start = Dataset._ensure_number_is_int(slice_obj.start)
                so_stop = Dataset._ensure_number_is_int(slice_obj.stop)
                if slice_obj.step is None:
                    so_step = 1
                else:
                    so_step = Dataset._ensure_number_is_int(slice_obj.step)
                slice_obj = slice(so_start, so_stop, so_step)
            return dataset[slice_obj]
        else:
            return dataset[slice_obj]

    @staticmethod
    def _default_base_slice_fn(slice_obj, dataset):
        if isinstance(slice_obj, numbers.Number):
            slice_obj = slice(slice_obj, slice_obj+1, 1)
        return Dataset._slice_dataset(slice_obj, dataset)

    @staticmethod
    def _default_slice_fn(slice_obj, sliced_dataset, dataset_size):
        if isinstance(slice_obj, numbers.Number):
            slice_obj = 0
        else:
            if slice_obj.stop > slice_obj.start:
                slice_size = slice_obj.stop - slice_obj.start
            else:
                slice_size = slice_obj.stop + dataset_size - slice_obj.start
            slice_obj = slice(0, slice_size, slice_obj.step)
        return Dataset._slice_dataset(slice_obj, sliced_dataset)

    def _get_base_item(self, slice_obj):
        base_dataset = self._slice_base_dataset(slice_obj, self._base_dataset)
        if self._trans_fn is not None:
            if self._elementwise_query_fn:
                vals = [self._trans_fn(base_dataset[i]) for i in range(base_dataset.size)]
                return ivy.Container.list_stack(vals, 0)
            return self._trans_fn(base_dataset)
        return base_dataset

    def _get_item_from_slice_objs(self, base_slice_obj, slice_obj):
        if isinstance(base_slice_obj, tuple):
            item = ivy.Container.list_join((self._get_base_item(base_slice_obj[0]),
                                            self._get_base_item(base_slice_obj[1])))
        else:
            item = self._get_base_item(base_slice_obj)
        return self._slice_dataset(slice_obj, item, self._size)

    def _wrap_slice_obj(self, slice_obj):
        if isinstance(slice_obj, numbers.Number):
            return slice_obj % self._size
        else:
            so_start_orig = slice_obj.start
            so_stop_orig = slice_obj.stop
            so_start_wrapped = so_start_orig % self._size
            if abs(so_stop_orig - so_start_orig - 1) < 1e-6:
                return slice(so_start_wrapped, so_start_wrapped + 1, 1)
            so_stop_wrapped = so_stop_orig % self._size
            if abs(so_stop_wrapped) < 1:
                so_stop_wrapped = self._size + so_stop_wrapped
            return slice(so_start_wrapped, so_stop_wrapped, 1)

    def _wrap_base_slice_obj(self, slice_obj):
        if isinstance(slice_obj, numbers.Number):
            return slice_obj
        elif slice_obj.stop < slice_obj.start:
            end_idx_0 = slice_obj.start + math.ceil(self._size - slice_obj.start)
            slice_obj_0 = slice(slice_obj.start, end_idx_0, 1)
            start_idx_1 = end_idx_0 - self._size
            slice_obj_1 = slice(start_idx_1, slice_obj.stop, 1)
            return slice_obj_0, slice_obj_1
        return slice_obj

    @staticmethod
    def _split_slice_obj(slice_obj, cache):
        if isinstance(slice_obj, numbers.Number):
            if slice_obj in cache:
                return [(True, slice_obj)]
            else:
                return [(False, slice_obj)]
        slice_objs = list()
        start = slice_obj.start
        for i in np.arange(slice_obj.start, slice_obj.stop, 1.):
            if i in cache:
                if i != start:
                    slice_objs.append((False, slice(start, i, 1)))
                slice_objs.append((True, slice(i, i+1, 1)))
                start = i + 1
        if start < slice_obj.stop:
            slice_objs.append((False, slice(start, slice_obj.stop, 1)))
        elif len(slice_objs) == 0:
            return [(False, slice_obj)]
        return slice_objs

    def _add_to_cache(self, so, item):
        if isinstance(so, numbers.Number):
            self._cache[so] = item
        else:
            for i in np.arange(so.start, so.stop-1e-3, 1.):
                self._cache[i] = Dataset._slice_dataset(i-so.start, item)

    def _end_multiprocessing(self):
        if self._num_processes > 1:
            try:
                for i, w in enumerate(self._workers):
                    self._slice_queues[i].put(None)
                    w.join(timeout=5.0)
                for q in self._slice_queues:
                    q.cancel_join_thread()
                    q.close()
                for q in self._output_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                for w in self._workers:
                    if w.is_alive():
                        w.terminate()
                del self._workers
                del self._slice_queues
                del self._output_queues

    def __del__(self):
        self._end_multiprocessing()

    def _get_item_after_cache_n_wrap(self, slice_obj):
        base_slice_obj = self._wrap_base_slice_obj(slice_obj)
        return self._get_item_from_slice_objs(base_slice_obj, slice_obj)

    def _get_item(self, slice_obj):
        slice_obj = self._wrap_slice_obj(slice_obj)
        split_slice_objs = self._split_slice_obj(slice_obj, self._cache)
        items = list()
        items_for_cache = list()
        sos_for_cache = list()
        for from_cache, so in split_slice_objs:
            if from_cache:
                so_key = so if isinstance(so, numbers.Number) else so.start
                items.append(self._cache[so_key])
                continue
            item = self._get_item_after_cache_n_wrap(so)
            if self._with_caching:
                sos_for_cache.append(so)
                items_for_cache.append(item)
            items.append(item)
        if self._cache_size > 0:
            for so, item in zip(sos_for_cache, items_for_cache):
                self._add_to_cache(so, item)
        if len(items) == 1:
            if isinstance(slice_obj, numbers.Number):
                return items[0]
            return items[0].map(lambda x, kc: x if isinstance(x, list) else [x])
        items_as_lists = [item.map(lambda x, kc: x if isinstance(x, list) else [x]) for item in items]
        return ivy.Container.list_join(items_as_lists)

    # Public #
    # -------#

    def __getitem__(self, slice_obj):
        if self._numpy_loading:
            ivy.set_framework('numpy')
        if self._num_processes < 2:
            ret = self._get_item(slice_obj)
            if self._numpy_loading:
                ivy.unset_framework()
            return ret
        if isinstance(slice_obj, numbers.Number):
            ret = self._get_item(slice_obj)
            if self._numpy_loading:
                ivy.unset_framework()
            return ret
        slice_size = int(round(slice_obj.stop - slice_obj.start))
        num_sub_slices = min(slice_size, self._num_processes)
        slice_points = np.linspace(slice_obj.start, slice_obj.stop, num_sub_slices+1)
        if Dataset._is_int(slice_obj.start) and Dataset._is_int(slice_obj.stop):
            slice_points = np.round(slice_points)
        sub_slices = [slice(slice_points[i], slice_points[i+1], 1.) for i in range(num_sub_slices)]
        offset = np.random.randint(0, self._num_processes)
        [self._slice_queues[int((i + offset) % self._num_processes)].put(sub_slice)
         for i, sub_slice in enumerate(sub_slices)]
        items_as_lists = [self._output_queues[int((i + offset) % self._num_processes)].get(timeout=5.0)
                          for i in range(num_sub_slices)]
        ret = ivy.Container.list_join(items_as_lists)
        if self._numpy_loading:
            ivy.unset_framework()
        return ivy.Container.list_join(items_as_lists)

    def map(self, name, map_func, num_processes=1, base_slice_fn=None, numpy_loading=None):
        return Dataset(base_dataset=self,
                       name=name,
                       size=self._size,
                       base_slice_fn=base_slice_fn,
                       trans_fn=map_func,
                       with_caching=self._with_caching,
                       cache_size=self._cache_size,
                       num_processes=num_processes,
                       numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading)

    def batch(self, name, batch_size, num_processes=1, numpy_loading=None):
        def batch_array(x, _):
            return [ivy.concatenate([ivy.expand_dims(item, 0) for item in x[i*batch_size:i*batch_size+batch_size]], 0)
                    for i in range(int(len(x)/batch_size))]

        def batch_cont(cont):
            return cont.map(batch_array)

        def base_slice_fn(slc_obj, dataset):
            if isinstance(slc_obj, numbers.Number):
                base_slice_obj =\
                    slice(int(round(batch_size * slc_obj)), int(round(batch_size * slc_obj + batch_size)), 1)
            else:
                so_start = int(round(batch_size * slc_obj.start))
                so_stop = int(round(batch_size * slc_obj.stop))
                base_slice_obj = slice(so_start, so_stop, 1)
            return Dataset._slice_dataset(base_slice_obj, dataset)

        return Dataset(base_dataset=self,
                       name=name,
                       size=float(self._size / batch_size),
                       base_slice_fn=base_slice_fn,
                       trans_fn=batch_cont,
                       elementwise_query_fn=False,
                       with_caching=self._with_caching,
                       cache_size=int(math.ceil(self._cache_size / batch_size)),
                       num_processes=num_processes,
                       numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading)

    def unbatch(self, name, num_processes=1, numpy_loading=None, cache_size=None, batch_sizes=None):

        unbatch_slice_dict = dict()
        slice_dict = dict()
        size_so_far = 0
        size = math.ceil(self._size)
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]*size
        for i in range(size):
            if batch_sizes is None:
                data = Dataset._slice_dataset(i, self)
                data_size = data.size
            else:
                data_size = batch_sizes[i]
            if i == size - 1 and self._size % 1 != 0:
                data_size = int(round(data_size * (self._size - math.floor(self._size))))
            for j in range(data_size):
                unbatch_slice_dict[size_so_far + j] = i
                slice_dict[size_so_far + j] = j
            size_so_far += data_size
        unrolled_size = size_so_far

        def base_slice_fn(slice_obj, dataset):
            if isinstance(slice_obj, numbers.Number):
                slice_obj = slice(slice_obj, slice_obj + 1, 1)
            so_start = unbatch_slice_dict[slice_obj.start]
            so_stop = unbatch_slice_dict[slice_obj.stop - 1] + 1
            so_stop = so_stop + 1 if so_stop == so_start else so_stop
            so = slice(so_start, so_stop, 1)
            return Dataset._slice_dataset(so, dataset)

        def unbatch_fn(cont):
            return cont.map(lambda x, kc: [c for o in [ivy.unstack(item, 0) for item in x] for c in o])

        def slice_fn(slice_obj, sliced_dataset, dataset_size):
            if isinstance(slice_obj, numbers.Number):
                return Dataset._slice_dataset(slice_dict[slice_obj], sliced_dataset)
            else:
                if slice_obj.stop > slice_obj.start:
                    slice_size = slice_obj.stop - slice_obj.start
                else:
                    slice_size = slice_obj.stop + unrolled_size - slice_obj.start
                so_start = slice_dict[slice_obj.start]
                so_stop = so_start + slice_size
                so = slice(so_start, so_stop, 1)
                return Dataset._slice_dataset(so, sliced_dataset)

        return Dataset(base_dataset=self,
                       name=name,
                       size=unrolled_size,
                       base_slice_fn=base_slice_fn,
                       trans_fn=unbatch_fn,
                       slice_fn=slice_fn,
                       elementwise_query_fn=False,
                       with_caching=self._with_caching,
                       cache_size=int(math.ceil(self._cache_size * unrolled_size / self._size))
                       if cache_size is None else cache_size,
                       num_processes=num_processes,
                       numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading)

    def shuffle(self, name, shuffle_buffer_size, num_processes=1, numpy_loading=None):
        if shuffle_buffer_size == 0:
            return self
        pre_shuffled = self.batch('pre_' + name,
                                  shuffle_buffer_size,
                                  num_processes=num_processes,
                                  numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading)
        shuffled = Dataset(base_dataset=pre_shuffled,
                           name=name,
                           size=pre_shuffled.size,
                           trans_fn=lambda cont: cont.shuffle(),
                           with_caching=self._with_caching,
                           cache_size=self._cache_size,
                           num_processes=num_processes,
                           numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading)
        post_shuffled = shuffled.unbatch('post_' + name,
                                         num_processes=num_processes,
                                         numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading,
                                         cache_size=self._cache_size,
                                         batch_sizes=shuffle_buffer_size)
        return post_shuffled

    def prefetch(self, name, buffer_size, num_processes=1, numpy_loading=None):

        # noinspection PyUnresolvedReferences
        def base_slice_fn(slc_obj, dataset):
            if isinstance(slc_obj, numbers.Number):
                so_start = slc_obj
                so_stop = slc_obj + 1 + buffer_size
            else:
                so_start = slc_obj.start
                so_stop = slc_obj.stop + buffer_size
            base_slice_obj = slice(so_start, so_stop, 1)
            return Dataset._slice_dataset(base_slice_obj, dataset)

        return Dataset(base_dataset=self,
                       name=name,
                       size=self._size,
                       base_slice_fn=base_slice_fn,
                       with_caching=self._with_caching,
                       cache_size=self._cache_size,
                       num_processes=num_processes,
                       numpy_loading=self._numpy_loading if numpy_loading is None else numpy_loading)

    def to_gpu(self, name, num_processes=1, gpu_idx=0):

        def item_to_gpu(x, _):
            return ivy.array(x, dev_str='gpu:' + str(gpu_idx))

        def cont_to_gpu(cont):
            return cont.map(item_to_gpu)

        return Dataset(base_dataset=self,
                       name=name,
                       size=self._size,
                       trans_fn=cont_to_gpu,
                       with_caching=self._with_caching,
                       cache_size=0,
                       num_processes=num_processes,
                       numpy_loading=False)

    # Getters #
    # --------#

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size
