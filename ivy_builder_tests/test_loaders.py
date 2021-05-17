# global
import os

# local
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.data_loaders.json_data_loader import JSONDataLoader
from ivy_builder.data_loaders.specs.json_data_loader_spec import JSONDataLoaderSpec


def test_json_loader(dev_str, f, call):

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, 'dataset')
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, 'containers'))

    dataset_spec = DatasetSpec(dataset_dirs, sequence_lengths=[2, 3, 2, 3, 3, 2])
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=3, window_size=2, num_sequences_to_use=6,
                                          num_training_sequences=3, preload_containers=True, float_strs=['depth'],
                                          uint8_strs=['rgb'])

    # data loader
    data_loader = JSONDataLoader(data_loader_spec)

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
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=3, window_size=3, num_sequences_to_use=6,
                                          num_training_sequences=3,
                                          unused_key_chains=['observations/image/ego/ego_cam_px/depth'],
                                          float_strs=['depth'], uint8_strs=['rgb'])
    data_loader = JSONDataLoader(data_loader_spec)

    train_batch = data_loader.get_next_batch('training')
    assert train_batch.actions.shape == (3, 3, 6)
    assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
    assert 'depth' not in train_batch.observations.image.ego.ego_cam_px
    valid_batch = data_loader.get_next_batch('validation')
    assert valid_batch.actions.shape == (3, 3, 6)
    assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 3, 32, 32, 3)
