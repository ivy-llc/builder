# global
import ivy
import math
import numbers


class Dataset:

    def __init__(self, dataset, name, size, base_slice_fn=None, trans_fn=None, slice_fn=None,
                 elementwise_query_fn=True):
        self._dataset = dataset
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
        # ToDo: add caching to prevent repeat reads

    # Private #
    # --------#

    @staticmethod
    def _ensure_number_is_int(val):
        if val % 1 > 1e-6:
            raise Exception('Trying to slice ivy Container with non-integer slice {}'.format(val))
        return int(round(val))

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
            return dataset.slice(slice_obj)
        else:
            return dataset[slice_obj]

    @staticmethod
    def _default_base_slice_fn(slice_obj, dataset):
        if isinstance(slice_obj, numbers.Number):
            slice_obj = slice(int(round(slice_obj)), int(round(slice_obj+1)), 1)
        else:
            slice_obj = slice(int(round(slice_obj.start)), int(round(slice_obj.stop)), int(round(slice_obj.step)))
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

    def _get_item(self, slice_obj):
        base_dataset = self._slice_base_dataset(slice_obj, self._dataset)
        if self._trans_fn is not None:
            if self._elementwise_query_fn:
                return ivy.Container.list_stack(
                    [self._trans_fn(base_dataset.slice(i))
                     for i in range(base_dataset.size)], 0)
            return self._trans_fn(base_dataset)
        return base_dataset

    # Public #
    # -------#

    def __getitem__(self, slice_obj):
        if isinstance(slice_obj, numbers.Number):
            slice_obj = slice_obj % self.size
            return self._slice_dataset(slice_obj, self._get_item(slice_obj), self._size)
        else:
            so_start = slice_obj.start % self.size
            so_stop = slice_obj.stop % self.size if slice_obj.stop != math.ceil(self.size) else slice_obj.stop
            slice_obj = slice(so_start, so_stop, 1)
            if so_stop < so_start:
                slice_obj_0 = slice(so_start, self.size, 1)
                slice_obj_1 = slice(0, so_stop, 1)
                item = ivy.Container.list_join((self._get_item(slice_obj_0), self._get_item(slice_obj_1)))
            else:
                item = self._get_item(slice_obj)
            return self._slice_dataset(slice_obj, item, self.size)

    def map(self, name, map_func, num_parallel_calls=1, base_slice_fn=None):
        return Dataset(dataset=self,
                       name=name,
                       size=self._size,
                       base_slice_fn=base_slice_fn,
                       trans_fn=map_func)

    def batch(self, name, batch_size):
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

        return Dataset(dataset=self,
                       name=name,
                       size=float(self._size / batch_size),
                       base_slice_fn=base_slice_fn,
                       trans_fn=batch_cont,
                       elementwise_query_fn=False)

    def unbatch(self, name):

        # ToDo: make this more efficient, without needing to traverse entire dataset during initialization
        #  this can be achieved with extra optional input for the leading sizes of each entry in the dataset
        unbatch_slice_dict = dict()
        slice_dict = dict()
        size_so_far = 0
        for i in range(self._size):
            data = Dataset._slice_dataset(i, self)
            for j in range(data.size):
                unbatch_slice_dict[size_so_far + j] = i
                slice_dict[size_so_far + j] = j
            size_so_far += data.size
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

        return Dataset(dataset=self,
                       name=name,
                       size=unrolled_size,
                       base_slice_fn=base_slice_fn,
                       trans_fn=unbatch_fn,
                       slice_fn=slice_fn,
                       elementwise_query_fn=False)

    def shuffle(self, name, shuffle_size):
        return Dataset(dataset=self,
                       name=name,
                       size=self._size,
                       trans_fn=lambda cont: cont.shuffle())

    def apply(self, name, transformation_func, size):
        # ToDo: implement
        return Dataset(dataset=self,
                       name=name,
                       size=self._size)

    def prefetch(self, name, buffer_size):
        # ToDo: implement
        return Dataset(dataset=self,
                       name=name,
                       size=self._size)

    # Getters #
    # --------#

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size
