# global
import os
import ivy
import json
import pytest
import numpy as np

# local
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.data_loaders.seq_data_loader import SeqDataLoader
from ivy_builder.data_loaders.specs.seq_data_loader_spec import SeqDataLoaderSpec


def test_seq_loader_multi_dev(dev_str, f):
    # seed
    ivy.seed(seed_value=0)
    np.random.seed(0)

    # devices
    dev_strs = list()
    dev_str0 = dev_str
    dev_strs.append(dev_str0)
    if "gpu" in dev_str and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev_str1 = dev_str[:-1] + str(idx)
        dev_strs.append(dev_str1)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, "dataset")
    dataset_dirs = DatasetDirs(
        dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, "containers")
    )

    dataset_spec = DatasetSpec(
        dataset_dirs, sequence_lengths=2, cont_fname_template="%06d_%06d.json"
    )
    data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=2,
        window_size=1,
        starting_idx=0,
        num_sequences=1,
        container_load_mode="preload",
        array_mode="hdf5",
        array_strs=["array"],
        float_strs=["depth"],
        uint8_strs=["rgb"],
        with_prefetching=False,
        shuffle_buffer_size=0,
        prefetch_to_devs=dev_strs,
        preshuffle_data=False,
    )

    # data loader
    data_loader = SeqDataLoader(data_loader_spec)

    # testing
    for i in range(5):
        # get training batch
        batch = data_loader.get_next_batch()

        # test cardinality
        assert batch.actions.shape == (2, 1, 6)
        assert batch.observations.image.ego.ego_cam_px.rgb.shape == (2, 1, 32, 32, 3)
        assert batch.observations.image.ego.ego_cam_px.rgb.shape == (2, 1, 32, 32, 3)
        assert batch["array"].data.shape == (2, 1, 3)

        # test values
        if len(dev_strs) == 1:
            assert batch.seq_info.length[0][0] == 2
            assert batch.seq_info.idx[0][0] == 0
            continue
        for j, ds in enumerate(dev_str):
            assert batch.seq_info.length[ds][0][0] == 2
            assert batch.seq_info.idx[ds][0][0] == j

    # delete
    data_loader.close()
    del data_loader


@pytest.mark.parametrize("container_load_mode", ["preload", "dynamic"])
@pytest.mark.parametrize("array_mode", ["hdf5", "pickled"])
@pytest.mark.parametrize("with_prefetching", [True, False])
@pytest.mark.parametrize("sequence_lengths", [1, 2])
@pytest.mark.parametrize("num_sequences", [1, 2, 3])
def test_seq_loader_fixed_seq_len(
    dev_str,
    f,
    container_load_mode,
    array_mode,
    with_prefetching,
    sequence_lengths,
    num_sequences,
):
    # seed
    ivy.seed(seed_value=0)
    np.random.seed(0)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, "dataset")
    dataset_dirs = DatasetDirs(
        dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, "containers")
    )

    dataset_spec = DatasetSpec(
        dataset_dirs,
        sequence_lengths=sequence_lengths,
        cont_fname_template="%06d_%06d.json",
    )
    data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=1,
        window_size=1,
        starting_idx=0,
        num_sequences=num_sequences,
        container_load_mode=container_load_mode,
        array_mode=array_mode,
        array_strs=["array"],
        float_strs=["depth"],
        uint8_strs=["rgb"],
        with_prefetching=with_prefetching,
        shuffle_buffer_size=0,
        preshuffle_data=False,
    )

    # data loader
    data_loader = SeqDataLoader(data_loader_spec)

    # testing
    for i in range(5):
        # get training batch
        batch = data_loader.get_next_batch()

        # test cardinality
        assert batch.actions.shape == (1, 1, 6)
        assert batch.observations.image.ego.ego_cam_px.rgb.shape == (1, 1, 32, 32, 3)
        assert batch.observations.image.ego.ego_cam_px.rgb.shape == (1, 1, 32, 32, 3)
        assert batch["array"].data.shape == (1, 1, 3)

        # test values
        if sequence_lengths == 2 and num_sequences == 1:
            assert batch.seq_info.length[0, 0] == 2
            assert batch.seq_info.idx[0, 0] == i % 2

    # delete
    data_loader.close()
    del data_loader


@pytest.mark.parametrize("container_load_mode", ["preload", "dynamic"])
@pytest.mark.parametrize("array_mode", ["hdf5", "pickled"])
@pytest.mark.parametrize("with_prefetching", [True, False])
@pytest.mark.parametrize("shuffle_buffer_size", [0, 2])
def test_seq_loader(
    dev_str, f, container_load_mode, array_mode, with_prefetching, shuffle_buffer_size
):
    # seed
    ivy.seed(seed_value=0)
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
    ds_dir = os.path.join(current_dir, "dataset")
    dataset_dirs = DatasetDirs(
        dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, "containers")
    )

    # data loader specifications
    dataset_spec = DatasetSpec(
        dataset_dirs,
        sequence_lengths=[2, 3, 2, 3, 3, 1],
        cont_fname_template="%06d_%06d.json",
    )
    train_data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=batch_size,
        window_size=window_size,
        starting_idx=0,
        num_sequences=3,
        container_load_mode=container_load_mode,
        array_mode=array_mode,
        array_strs=["array"],
        float_strs=["depth"],
        uint8_strs=["rgb"],
        with_prefetching=with_prefetching,
        shuffle_buffer_size=shuffle_buffer_size,
        preshuffle_data=False,
    )
    valid_data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=batch_size,
        window_size=window_size,
        starting_idx=3,
        num_sequences=3,
        container_load_mode=container_load_mode,
        array_mode=array_mode,
        array_strs=["array"],
        float_strs=["depth"],
        uint8_strs=["rgb"],
        with_prefetching=with_prefetching,
        shuffle_buffer_size=shuffle_buffer_size,
        preshuffle_data=False,
    )

    # training data loader
    train_data_loader = SeqDataLoader(train_data_loader_spec)

    # validation data loader
    valid_data_loader = SeqDataLoader(valid_data_loader_spec)

    # testing
    for i in range(5):
        # get training batch
        train_batch = train_data_loader.get_next_batch()

        # test cardinality
        assert train_batch.actions.shape == (3, 2, 6)
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (
            3,
            2,
            32,
            32,
            3,
        )
        assert train_batch.observations.image.ego.ego_cam_px.rgb.shape == (
            3,
            2,
            32,
            32,
            3,
        )
        assert train_batch["array"].data.shape == (3, 2, 3)

        # test values
        window_idxs = [
            i % 4 for i in list(range(i * batch_size, i * batch_size + batch_size))
        ]
        seq_idxs = [np.argmax(wi < cum_seq_wind_sizes_train) for wi in window_idxs]
        seq_lens = [seq_lengths_train[si] for si in seq_idxs]
        padded_seq_lens = [padded_seq_lengths_train[si] for si in seq_idxs]
        unpadded_mask = [sl == psl for sl, psl in zip(seq_lens, padded_seq_lens)]
        in_seq_win_idxs = [
            (wi - cum_seq_wind_sizes_train[si - 1]) if si != 0 else wi
            for si, wi in zip(seq_idxs, window_idxs)
        ]
        if shuffle_buffer_size == 0:
            assert np.allclose(
                ivy.to_numpy(train_batch.seq_info.length),
                np.tile(
                    np.asarray(padded_seq_lens, dtype=np.float32).reshape(-1, 1), (1, 2)
                ),
            )
            assert np.allclose(
                ivy.to_numpy(train_batch.seq_info.idx),
                np.concatenate(
                    (
                        np.asarray(in_seq_win_idxs, dtype="float32").reshape(-1, 1),
                        np.asarray(in_seq_win_idxs, dtype="float32").reshape(-1, 1)
                        + np.asarray(unpadded_mask, dtype="float32").reshape(-1, 1),
                    ),
                    -1,
                ),
            )

        # get validation batch
        valid_batch = valid_data_loader.get_next_batch()

        # test cardinality
        assert valid_batch.actions.shape == (3, 2, 6)
        assert valid_batch.observations.image.ego.ego_cam_px.rgb.shape == (
            3,
            2,
            32,
            32,
            3,
        )
        assert valid_batch["array"].data.shape == (3, 2, 3)

        # test values
        window_idxs = [
            i % 5 for i in list(range(i * batch_size, i * batch_size + batch_size))
        ]
        seq_idxs = [np.argmax(wi < cum_seq_wind_sizes_valid) for wi in window_idxs]
        seq_lens = [seq_lengths_valid[si] for si in seq_idxs]
        padded_seq_lens = [padded_seq_lengths_valid[si] for si in seq_idxs]
        unpadded_mask = [sl == psl for sl, psl in zip(seq_lens, padded_seq_lens)]
        in_seq_win_idxs = [
            (wi - cum_seq_wind_sizes_valid[si - 1]) if si != 0 else wi
            for si, wi in zip(seq_idxs, window_idxs)
        ]
        if shuffle_buffer_size == 0:
            assert np.allclose(
                ivy.to_numpy(valid_batch.seq_info.length),
                np.tile(
                    np.asarray(padded_seq_lens, dtype=np.float32).reshape(-1, 1), (1, 2)
                ),
            )
            assert np.allclose(
                ivy.to_numpy(valid_batch.seq_info.idx),
                np.concatenate(
                    (
                        np.asarray(in_seq_win_idxs, dtype="float32").reshape(-1, 1),
                        np.asarray(in_seq_win_idxs, dtype="float32").reshape(-1, 1)
                        + np.asarray(unpadded_mask, dtype="float32").reshape(-1, 1),
                    ),
                    -1,
                ),
            )

    # delete
    train_data_loader.close()
    del train_data_loader
    valid_data_loader.close()
    del valid_data_loader

    # test keychain pruning, no container pre-loading
    data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=3,
        window_size=2,
        starting_idx=0,
        num_sequences=3,
        container_load_mode=container_load_mode,
        array_mode=array_mode,
        shuffle_buffer_size=shuffle_buffer_size,
        unused_key_chains=["observations/image/ego/ego_cam_px/depth"],
        array_strs=["array"],
        float_strs=["depth"],
        uint8_strs=["rgb"],
        with_prefetching=with_prefetching,
        preshuffle_data=False,
    )
    data_loader = SeqDataLoader(data_loader_spec)

    # get training batch
    batch = data_loader.get_next_batch()

    # test cardinality
    assert batch.actions.shape == (3, 2, 6)
    assert batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
    assert batch.observations.image.ego.ego_cam_px.rgb.shape == (3, 2, 32, 32, 3)
    assert batch["array"].data.shape == (3, 2, 3)

    # test removed key chain
    assert "depth" not in batch.observations.image.ego.ego_cam_px

    # delete
    data_loader.close()
    del data_loader


@pytest.mark.parametrize("container_load_mode", ["preload", "dynamic"])
@pytest.mark.parametrize("array_mode", ["hdf5", "pickled"])
@pytest.mark.parametrize("with_prefetching", [True, False])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize(
    "containers_to_skip",
    [
        [(0, 0), (0, 1), (1, 1), (5, 0)],
        [(0, 1), (1, 1), (5, 0)],
        [(0, 0), (2, 1), (1, 2), (3, 1)],
        [(1, None), (3, None), (4, 0)],
        [(None, 0), (None, 2), (3, 1)],
    ],
)
def test_seq_loader_containers_to_skip(
    dev_str,
    f,
    container_load_mode,
    array_mode,
    with_prefetching,
    batch_size,
    containers_to_skip,
):
    # seed
    ivy.seed(seed_value=0)
    np.random.seed(0)

    # dataset dir
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ds_dir = os.path.join(current_dir, "dataset")
    dataset_dirs = DatasetDirs(
        dataset_dir=ds_dir, containers_dir=os.path.join(ds_dir, "containers")
    )

    dataset_spec = DatasetSpec(
        dataset_dirs,
        sequence_lengths=[2, 3, 2, 3, 3, 1],
        cont_fname_template="%06d_%06d.json",
    )
    data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=batch_size,
        window_size=1,
        starting_idx=0,
        container_load_mode=container_load_mode,
        array_mode=array_mode,
        num_sequences=6,
        array_strs=["array"],
        float_strs=["depth"],
        uint8_strs=["rgb"],
        preshuffle_data=False,
        with_prefetching=with_prefetching,
        num_workers=1,
        containers_to_skip=containers_to_skip,
    )

    seq_idx_length_and_idxs = [
        [0, 2, 0],
        [0, 2, 1],
        [1, 3, 0],
        [1, 3, 1],
        [1, 3, 2],
        [2, 2, 0],
        [2, 2, 1],
        [3, 3, 0],
        [3, 3, 1],
        [3, 3, 2],
        [4, 3, 0],
        [4, 3, 1],
        [4, 3, 2],
        [5, 1, 0],
    ]

    def _skip(seq_idx_length_and_idx):
        seq_idx_, _, idx_ = seq_idx_length_and_idx
        if (
            (seq_idx_, None) in containers_to_skip
            or (None, idx_) in containers_to_skip
            or (seq_idx_, idx_) in containers_to_skip
        ):
            return True
        return False

    seq_idxs = [i[0] for i in seq_idx_length_and_idxs if not _skip(i)]
    lengths = [i[1] for i in seq_idx_length_and_idxs if not _skip(i)]
    idxs = [i[2] for i in seq_idx_length_and_idxs if not _skip(i)]

    # data loader
    data_loader = SeqDataLoader(data_loader_spec)

    # testing
    for i in range(7):
        # get ground truth
        idx = list()
        length = list()
        seq_idx = list()
        for j in range(batch_size):
            idx.append(idxs[(i * batch_size + j) % len(idxs)])
            length.append(lengths[(i * batch_size + j) % len(lengths)])
            seq_idx.append(seq_idxs[(i * batch_size + j) % len(seq_idxs)])

        # get training batch
        batch = data_loader.get_next_batch()

        # test seq_info
        assert np.array_equal(
            ivy.to_numpy(batch.seq_info.idx), np.expand_dims(np.asarray(idx), -1)
        )
        assert np.array_equal(
            ivy.to_numpy(batch.seq_info.length), np.expand_dims(np.asarray(length), -1)
        )
        assert np.array_equal(
            ivy.to_numpy(batch.seq_info.seq_idx),
            np.expand_dims(np.asarray(seq_idx), -1),
        )

    # delete
    data_loader.close()
    del data_loader


@pytest.mark.parametrize("with_prefetching", [True, False])
@pytest.mark.parametrize("shuffle_buffer_size", [0, 2])
def test_seq_loader_wo_cont_load(dev_str, f, with_prefetching, shuffle_buffer_size):
    # seed
    ivy.seed(seed_value=0)
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
    ds_dir = os.path.join(current_dir, "dataset")
    dataset_dirs = DatasetDirs(dataset_dir=ds_dir)

    # custom init function
    def custom_init_fn(self):
        alternative_data_dir = os.path.join(
            self._spec.dataset_spec.dirs.dataset_dir, "containers_alternative/"
        )
        actions_fpath = os.path.join(alternative_data_dir, "actions.json")
        with open(actions_fpath, "r") as file:
            self._actions_dict = json.loads(file.read())

    # custom load function
    def custom_container_load_fn(self, cont):
        new_cont = ivy.Container()
        all_idxs = cont.idx_map.to_idxs()

        actions_seqs_list = list()

        seq_idxs_seqs_list = list()
        idxs_seqs_list = list()
        lengths_seqs_list = list()

        for seq in all_idxs:
            action_arrays_list = list()

            seq_idx_arrays_list = list()
            idx_arrays_list = list()

            found_end = False
            j = 0
            idx = 0
            last_idx = 0
            seq_idx = seq[0][0]

            for j, (_, idx) in enumerate(seq):
                if not ivy.exists(idx) and not found_end:
                    found_end = True
                    last_idx = j - 1
                if found_end:
                    idx = last_idx

                action_as_list = self._actions_dict[str(seq_idx)][str(idx)]
                action_arrays_list.append(ivy.array(action_as_list, dtype="float32")[0])

                seq_idx_arrays_list.append(ivy.array([seq_idx], dtype="float32"))
                idx_arrays_list.append(ivy.array([idx], dtype="float32"))
            length_arrays_list = [
                ivy.array([last_idx + 1 if found_end else idx + 1], dtype="float32")
            ] * (j + 1)

            action_arrays = ivy.concat(action_arrays_list, axis=0)
            actions_seqs_list.append(action_arrays)

            seq_idx_arrays = ivy.concat(seq_idx_arrays_list, axis=0)
            seq_idxs_seqs_list.append(seq_idx_arrays)
            idx_arrays = ivy.concat(idx_arrays_list, axis=0)
            idxs_seqs_list.append(idx_arrays)
            length_arrays = ivy.concat(length_arrays_list, axis=0)
            lengths_seqs_list.append(length_arrays)

        new_cont.actions = actions_seqs_list

        new_cont.seq_info = ivy.Container()
        new_cont.seq_info.seq_idx = seq_idxs_seqs_list
        new_cont.seq_info.idx = idxs_seqs_list
        new_cont.seq_info.length = lengths_seqs_list

        return new_cont

    # data loader specifications
    dataset_spec = DatasetSpec(
        dataset_dirs,
        sequence_lengths=[2, 3, 2, 3, 3, 1],
        cont_fname_template="%06d_%06d.json",
    )
    train_data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=batch_size,
        window_size=window_size,
        starting_idx=0,
        num_sequences=3,
        array_strs=["array"],
        with_prefetching=with_prefetching,
        container_load_mode="custom",
        shuffle_buffer_size=shuffle_buffer_size,
        preshuffle_data=False,
        custom_init_fn=custom_init_fn,
        custom_container_load_fn=custom_container_load_fn,
    )
    valid_data_loader_spec = SeqDataLoaderSpec(
        dataset_spec,
        batch_size=batch_size,
        window_size=window_size,
        starting_idx=3,
        num_sequences=3,
        array_strs=["array"],
        with_prefetching=with_prefetching,
        container_load_mode="custom",
        shuffle_buffer_size=shuffle_buffer_size,
        preshuffle_data=False,
        custom_init_fn=custom_init_fn,
        custom_container_load_fn=custom_container_load_fn,
    )

    # training data loader
    train_data_loader = SeqDataLoader(train_data_loader_spec)

    # validation data loader
    valid_data_loader = SeqDataLoader(valid_data_loader_spec)

    # testing
    for i in range(5):
        # get training batch
        train_batch = train_data_loader.get_next_batch()

        # test cardinality
        assert train_batch.actions.shape == (3, 2, 6)

        # test values
        window_idxs = [
            i % 4 for i in list(range(i * batch_size, i * batch_size + batch_size))
        ]
        seq_idxs = [np.argmax(wi < cum_seq_wind_sizes_train) for wi in window_idxs]
        seq_lens = [seq_lengths_train[si] for si in seq_idxs]
        padded_seq_lens = [padded_seq_lengths_train[si] for si in seq_idxs]
        unpadded_mask = [sl == psl for sl, psl in zip(seq_lens, padded_seq_lens)]
        in_seq_win_idxs = [
            (wi - cum_seq_wind_sizes_train[si - 1]) if si != 0 else wi
            for si, wi in zip(seq_idxs, window_idxs)
        ]
        if shuffle_buffer_size == 0:
            assert np.allclose(
                ivy.to_numpy(train_batch.seq_info.length),
                np.tile(np.array(padded_seq_lens).reshape(-1, 1), (1, 2)),
            )
            assert np.allclose(
                ivy.to_numpy(train_batch.seq_info.idx),
                np.concatenate(
                    (
                        np.array(in_seq_win_idxs).reshape(-1, 1),
                        np.array(in_seq_win_idxs).reshape(-1, 1)
                        + np.array(unpadded_mask).reshape(-1, 1),
                    ),
                    -1,
                ),
            )

        # get validation batch
        valid_batch = valid_data_loader.get_next_batch()

        # test cardinality
        assert valid_batch.actions.shape == (3, 2, 6)

        # test values
        window_idxs = [
            i % 5 for i in list(range(i * batch_size, i * batch_size + batch_size))
        ]
        seq_idxs = [np.argmax(wi < cum_seq_wind_sizes_valid) for wi in window_idxs]
        seq_lens = [seq_lengths_valid[si] for si in seq_idxs]
        padded_seq_lens = [padded_seq_lengths_valid[si] for si in seq_idxs]
        unpadded_mask = [sl == psl for sl, psl in zip(seq_lens, padded_seq_lens)]
        in_seq_win_idxs = [
            (wi - cum_seq_wind_sizes_valid[si - 1]) if si != 0 else wi
            for si, wi in zip(seq_idxs, window_idxs)
        ]
        if shuffle_buffer_size == 0:
            assert np.allclose(
                ivy.to_numpy(valid_batch.seq_info.length),
                np.tile(np.array(seq_lens).reshape(-1, 1), (1, 2)),
            )
            assert np.allclose(
                ivy.to_numpy(valid_batch.seq_info.idx),
                np.concatenate(
                    (
                        np.array(in_seq_win_idxs).reshape(-1, 1),
                        np.array(in_seq_win_idxs).reshape(-1, 1)
                        + np.array(unpadded_mask).reshape(-1, 1),
                    ),
                    -1,
                ),
            )

    # delete
    train_data_loader.close()
    del train_data_loader
    valid_data_loader.close()
    del valid_data_loader
