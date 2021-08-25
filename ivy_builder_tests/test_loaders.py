# global
import os
import ivy
import pytest
import numpy as np

# local
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.data_loaders.json_data_loader import JSONDataLoader
from ivy_builder.data_loaders.specs.json_data_loader_spec import JSONDataLoaderSpec


@pytest.mark.parametrize(
    "preload_containers", [True, False])
@pytest.mark.parametrize(
    "array_mode", ['hdf5', 'pickled'])
@pytest.mark.parametrize(
    "with_prefetching", [True, False])
@pytest.mark.parametrize(
    "shuffle_buffer_size", [0, 2])
def test_json_loader_fixed_seq_len(dev_str, f, call, preload_containers, array_mode, with_prefetching,
                                   shuffle_buffer_size):

    # seed
    f.seed(0)
    np.random.seed(0)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, 'dataset')
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, 'containers'))

    dataset_spec = DatasetSpec(dataset_dirs, sequence_lengths=2)
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=1, window_size=1, starting_idx=0,
                                          num_sequences=1, cont_fname_template='%06d_%06d.json',
                                          preload_containers=preload_containers, array_mode=array_mode,
                                          array_strs=['array'], float_strs=['depth'], uint8_strs=['rgb'],
                                          with_prefetching=with_prefetching, shuffle_buffer_size=shuffle_buffer_size,
                                          preshuffle_data=False)

    # data loader
    data_loader = JSONDataLoader(data_loader_spec)

    # testing
    for i in range(5):

        # get training batch
        batch = data_loader.get_next_batch()

        # test cardinality
        assert batch.actions.shape == (1, 1, 6)
        assert batch.observations.image.ego.ego_cam_px.rgb.shape == (1, 1, 32, 32, 3)
        assert batch.observations.image.ego.ego_cam_px.rgb.shape == (1, 1, 32, 32, 3)
        assert batch.array.data.shape == (1, 1, 3)

        # test values
        assert batch.seq_info.length[0, 0] == 2
        if shuffle_buffer_size == 0:
            assert batch.seq_info.idx[0, 0] == i % 2
        else:
            idx = batch.seq_info.idx[0, 0]
            assert idx in [0, 1]

    # delete
    data_loader.close()
    del data_loader


@pytest.mark.parametrize(
    "preload_containers", [True, False])
@pytest.mark.parametrize(
    "array_mode", ['hdf5', 'pickled'])
@pytest.mark.parametrize(
    "with_prefetching", [True, False])
@pytest.mark.parametrize(
    "shuffle_buffer_size", [0, 2])
def test_json_loader(dev_str, f, call, preload_containers, array_mode, with_prefetching, shuffle_buffer_size):

    # seed
    f.seed(0)
    np.random.seed(0)

    # config
    batch_size = 3
    window_size = 2
    _seq_lengths = [2, 3, 2, 3, 3, 1]
    _padded_seq_lenghts = [max(sl, window_size) for sl in _seq_lengths]
    _seq_wind_sizes = [psl - window_size + 1 for psl in _padded_seq_lenghts]

    seq_lengths_train = _seq_lengths[0:3]
    padded_seq_lengths_train = _padded_seq_lenghts[0:3]
    seq_wind_sizes_train = _seq_wind_sizes[0:3]
    cum_seq_wind_sizes_train = np.cumsum(seq_wind_sizes_train)

    seq_lengths_valid = _seq_lengths[3:6]
    padded_seq_lengths_valid = _padded_seq_lenghts[3:6]
    seq_wind_sizes_valid = _seq_wind_sizes[3:6]
    cum_seq_wind_sizes_valid = np.cumsum(seq_wind_sizes_valid)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, 'dataset')
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, 'containers'))

    # data loader specifications
    dataset_spec = DatasetSpec(dataset_dirs, sequence_lengths=[2, 3, 2, 3, 3, 1])
    train_data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=batch_size, window_size=window_size,
                                                starting_idx=0, num_sequences=3, cont_fname_template='%06d_%06d.json',
                                                preload_containers=preload_containers, array_mode=array_mode,
                                                array_strs=['array'], float_strs=['depth'], uint8_strs=['rgb'],
                                                with_prefetching=with_prefetching,
                                                shuffle_buffer_size=shuffle_buffer_size, preshuffle_data=False)
    valid_data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=batch_size, window_size=window_size,
                                                starting_idx=3, num_sequences=3, cont_fname_template='%06d_%06d.json',
                                                preload_containers=preload_containers, array_mode=array_mode,
                                                array_strs=['array'], float_strs=['depth'], uint8_strs=['rgb'],
                                                with_prefetching=with_prefetching,
                                                shuffle_buffer_size=shuffle_buffer_size, preshuffle_data=False)

    # training data loader
    train_data_loader = JSONDataLoader(train_data_loader_spec)

    # validation data loader
    valid_data_loader = JSONDataLoader(valid_data_loader_spec)

    # testing
    for i in range(5):

        # get training batch
        train_batch = train_data_loader.get_next_batch()

        # test cardinality
        assert train_batch.actions.shape == (3, 2, 6)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert train_batch.array.data.shape == (3, 2, 3)

        # test values
        window_idxs = [i % 4 for i in list(range(i*batch_size, i*batch_size+batch_size))]
        seq_idxs = [np.argmax(wi < cum_seq_wind_sizes_train) for wi in window_idxs]
        seq_lens = [seq_lengths_train[si] for si in seq_idxs]
        padded_seq_lens = [padded_seq_lengths_train[si] for si in seq_idxs]
        unpadded_mask = [sl == psl for sl, psl in zip(seq_lens, padded_seq_lens)]
        in_seq_win_idxs = [(wi - cum_seq_wind_sizes_train[si-1])
                           if si != 0 else wi for si, wi in zip(seq_idxs, window_idxs)]
        if shuffle_buffer_size == 0:
            assert np.allclose(ivy.to_numpy(train_batch.seq_info.length),
                               np.tile(np.array(padded_seq_lens).reshape(-1, 1), (1, 2)))
            assert np.allclose(ivy.to_numpy(train_batch.seq_info.idx),
                               np.concatenate((np.array(in_seq_win_idxs).reshape(-1, 1),
                                               np.array(in_seq_win_idxs).reshape(-1, 1) +
                                               np.array(unpadded_mask).reshape(-1, 1)), -1))

        # get validation batch
        valid_batch = valid_data_loader.get_next_batch()

        # test cardinality
        assert valid_batch.actions.shape == (3, 2, 6)
        assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
        assert valid_batch.array.data.shape == (3, 2, 3)

        # test values
        window_idxs = [i % 5 for i in list(range(i*batch_size, i*batch_size+batch_size))]
        seq_idxs = [np.argmax(wi < cum_seq_wind_sizes_valid) for wi in window_idxs]
        seq_lens = [seq_lengths_valid[si] for si in seq_idxs]
        padded_seq_lens = [padded_seq_lengths_valid[si] for si in seq_idxs]
        unpadded_mask = [sl == psl for sl, psl in zip(seq_lens, padded_seq_lens)]
        in_seq_win_idxs = [(wi - cum_seq_wind_sizes_valid[si-1])
                           if si != 0 else wi for si, wi in zip(seq_idxs, window_idxs)]
        if shuffle_buffer_size == 0:
            assert np.allclose(ivy.to_numpy(valid_batch.seq_info.length),
                               np.tile(np.array(padded_seq_lens).reshape(-1, 1), (1, 2)))
            assert np.allclose(ivy.to_numpy(valid_batch.seq_info.idx),
                               np.concatenate((np.array(in_seq_win_idxs).reshape(-1, 1),
                                               np.array(in_seq_win_idxs).reshape(-1, 1) +
                                               np.array(unpadded_mask).reshape(-1, 1)), -1))

    # delete
    train_data_loader.close()
    del train_data_loader
    valid_data_loader.close()
    del valid_data_loader

    # test keychain pruning, no container pre-loading
    data_loader_spec = JSONDataLoaderSpec(dataset_spec, batch_size=3, window_size=2, starting_idx=0,
                                          num_sequences=3, cont_fname_template='%06d_%06d.json',
                                          preload_containers=preload_containers, array_mode=array_mode,
                                          shuffle_buffer_size=shuffle_buffer_size,
                                          unused_key_chains=['observations/image/ego/ego_cam_px/depth'],
                                          array_strs=['array'], float_strs=['depth'], uint8_strs=['rgb'],
                                          with_prefetching=with_prefetching, preshuffle_data=False)
    data_loader = JSONDataLoader(data_loader_spec)

    # get training batch
    batch = data_loader.get_next_batch()

    # test cardinality
    assert batch.actions.shape == (3, 2, 6)
    assert batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
    assert batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
    assert batch.array.data.shape == (3, 2, 3)

    # test removed key chain
    assert 'depth' not in batch.observations.image.ego.ego_cam_px

    # delete
    data_loader.close()
    del data_loader
