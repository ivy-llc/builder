# global
import os
import pytest

# local
import ivy.tensorflow
import ivy_tests.helpers as helpers

from ivy_builder.specs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs import DataLoaderSpec

from ivy_builder.data_loaders.tf_data_loader import TFDataLoader

DataLoaders = {ivy.tensorflow: TFDataLoader}


def test_loaders(dev_str, f, call):

    if call not in [helpers.tf_call, helpers.tf_graph_call]:
        # ivy builder currently onlu supports tensorflow
        pytest.skip()

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, 'dataset')
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, 'containers'))

    dataset_spec = DatasetSpec(dataset_dirs, sequence_lengths=[2, 3, 2, 3, 3, 2])
    data_loader_spec = DataLoaderSpec(dataset_spec, None, shuffle_buffer_size=0, batch_size=3,
                                      window_size=2, num_sequences_to_use=6, num_training_sequences=3,
                                      post_proc_fn=None, prefetch_to_gpu=False, preload_containers=True)

    # data loader
    data_loader_class = DataLoaders[f]
    data_loader = data_loader_class(data_loader_spec)

    # testing
    for i in range(2):
        train_batch = data_loader.get_next_batch('training')
        assert train_batch.actions.shape == (3, 2, 6)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        valid_batch = data_loader.get_next_batch('validation')
        assert valid_batch.actions.shape == (3, 2, 6)
        assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)

    # test keychain pruning, no container pre-loading, and padded windowing
    data_loader_spec = DataLoaderSpec(dataset_spec, None, shuffle_buffer_size=0, batch_size=3,
                                      window_size=3, num_sequences_to_use=6, num_training_sequences=3,
                                      post_proc_fn=None, prefetch_to_gpu=False, preload_containers=False,
                                      unused_key_chains=['observations/image/ego/ego_cam_px/depth'])
    data_loader = data_loader_class(data_loader_spec)

    train_batch = data_loader.get_next_batch('training')
    assert train_batch.actions.shape == (3, 3, 6)
    assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert 'depth' not in train_batch.observations.image.ego.ego_cam_px
    valid_batch = data_loader.get_next_batch('validation')
    assert valid_batch.actions.shape == (3, 3, 6)
    assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
