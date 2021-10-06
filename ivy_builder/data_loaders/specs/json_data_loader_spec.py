# global
import ivy

# local
from ivy_builder.specs.spec import locals_to_kwargs
from ivy_builder.specs.data_loader_spec import DataLoaderSpec


class JSONDataLoaderSpec(DataLoaderSpec):

    def __init__(self, dataset_spec, batch_size, starting_idx, num_sequences, window_size=1,
                 num_workers=1, cache_size=0, unused_key_chains=None, preload_containers=False, preshuffle_data=True,
                 shuffle_buffer_size=0, with_prefetching=True, queue_timeout=None, post_proc_fn=None,
                 prefetch_to_devs='gpu:0', single_pass=False, array_strs=None, float_strs=None, uint8_strs=None,
                 custom_img_strs=None, custom_img_fns=None, custom_strs=None, custom_fns=None, array_mode='pickled',
                 load_gray_as_rgb=True, containers_to_skip=None, **kwargs):

        kw = locals_to_kwargs(locals())

        unused_key_chains = ivy.default(unused_key_chains, [])
        array_strs = ivy.default(array_strs, [])
        float_strs = ivy.default(float_strs, [])
        uint8_strs = ivy.default(uint8_strs, [])
        custom_img_strs = ivy.default(custom_img_strs, [[]])
        custom_img_fns = ivy.default(custom_img_fns, [])
        custom_strs = ivy.default(custom_strs, [[]])
        custom_fns = ivy.default(custom_fns, [])
        containers_to_skip = ivy.default(containers_to_skip, [])
        prefetch_to_devs = prefetch_to_devs if ivy.gpu_is_available() or isinstance(prefetch_to_devs, list) else False

        super(JSONDataLoaderSpec, self).__init__(dataset_spec,
                                                 batch_size=batch_size,
                                                 window_size=window_size,
                                                 starting_idx=starting_idx,
                                                 num_sequences=num_sequences,
                                                 num_workers=num_workers,
                                                 cache_size=cache_size,
                                                 unused_key_chains=unused_key_chains,
                                                 preload_containers=preload_containers,
                                                 preshuffle_data=preshuffle_data,
                                                 shuffle_buffer_size=shuffle_buffer_size,
                                                 with_prefetching=with_prefetching,
                                                 post_proc_fn=post_proc_fn,
                                                 prefetch_to_devs=prefetch_to_devs,
                                                 single_pass=single_pass,
                                                 array_strs=array_strs,
                                                 float_strs=float_strs,
                                                 uint8_strs=uint8_strs,
                                                 custom_img_strs=custom_img_strs,
                                                 custom_img_fns=custom_img_fns,
                                                 custom_strs=custom_strs,
                                                 custom_fns=custom_fns,
                                                 array_mode=array_mode,
                                                 load_gray_as_rgb=load_gray_as_rgb,
                                                 containers_to_skip=containers_to_skip,
                                                 **kwargs)
        self.queue_timeout = ivy.default(queue_timeout, ivy.queue_timeout())  # conflicts with ivy.Container argument

        self._kwargs = kw

    @property
    def kwargs(self):
        return self._kwargs
