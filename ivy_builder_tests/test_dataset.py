# global
import ivy
import pytest
import numpy as np

# local
from ivy_builder.dataset import Dataset


class TestQueries:

    def _init(self, array_shape):
        x = [ivy.array(0), ivy.array(1), ivy.array(2),
             ivy.array(3), ivy.array(4), ivy.array(5),
             ivy.array(6), ivy.array(7), ivy.array(8)]
        self._x = [ivy.reshape(item, array_shape) for item in x]
        dataset_container = ivy.Container({'x': self._x})
        self._dataset = Dataset(dataset_container, 'base', dataset_container.size)

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_single(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert list(self._dataset[0].x.shape) == array_shape
        assert list(self._dataset[4].x.shape) == array_shape
        assert list(self._dataset[8].x.shape) == array_shape

        assert np.allclose(ivy.to_numpy(self._dataset[0].x), ivy.to_numpy(self._x[0]))
        assert np.allclose(ivy.to_numpy(self._dataset[4].x), ivy.to_numpy(self._x[4]))
        assert np.allclose(ivy.to_numpy(self._dataset[8].x), ivy.to_numpy(self._x[8]))

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_single_wrapped(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert list(self._dataset[9].x.shape) == array_shape
        assert list(self._dataset[11].x.shape) == array_shape
        assert list(self._dataset[-1].x.shape) == array_shape
        assert list(self._dataset[-2].x.shape) == array_shape

        assert np.allclose(ivy.to_numpy(self._dataset[9].x), ivy.to_numpy(self._x[0]))
        assert np.allclose(ivy.to_numpy(self._dataset[11].x), ivy.to_numpy(self._x[2]))
        assert np.allclose(ivy.to_numpy(self._dataset[-1].x), ivy.to_numpy(self._x[-1]))
        assert np.allclose(ivy.to_numpy(self._dataset[-2].x), ivy.to_numpy(self._x[-2]))

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_slice(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert len(self._dataset[0:3].x) == 3
        assert list(self._dataset[0:3].x[0].shape) == array_shape
        assert len(self._dataset[3:6].x) == 3
        assert list(self._dataset[3:6].x[0].shape) == array_shape
        assert len(self._dataset[6:9].x) == 3
        assert list(self._dataset[6:9].x[0].shape) == array_shape

        assert np.allclose(ivy.to_numpy(self._dataset[0:3].x[0]), ivy.to_numpy(self._x[0]))
        assert np.allclose(ivy.to_numpy(self._dataset[3:6].x[1]), ivy.to_numpy(self._x[4]))
        assert np.allclose(ivy.to_numpy(self._dataset[6:9].x[2]), ivy.to_numpy(self._x[8]))

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_slice_wrapped(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert len(self._dataset[-1:1].x) == 2
        assert list(self._dataset[-1:1].x[0].shape) == array_shape
        assert len(self._dataset[-1:1].x) == 2
        assert list(self._dataset[-1:1].x[1].shape) == array_shape
        assert len(self._dataset[9:11].x) == 2
        assert list(self._dataset[9:11].x[0].shape) == array_shape
        assert len(self._dataset[9:11].x) == 2
        assert list(self._dataset[9:11].x[0].shape) == array_shape

        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[0]), ivy.to_numpy(self._x[8]))
        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[1]), ivy.to_numpy(self._x[0]))
        assert np.allclose(ivy.to_numpy(self._dataset[9:11].x[0]), ivy.to_numpy(self._x[0]))
        assert np.allclose(ivy.to_numpy(self._dataset[9:11].x[1]), ivy.to_numpy(self._x[1]))


class TestBatch:

    def _init(self, array_shape):
        x = [ivy.array(0), ivy.array(1), ivy.array(2), ivy.array(3), ivy.array(4),
             ivy.array(5), ivy.array(6), ivy.array(7), ivy.array(8), ivy.array(9)]
        self._x = [ivy.reshape(item, array_shape) for item in x]
        dataset_container = ivy.Container({'x': self._x})
        dataset = Dataset(dataset_container, 'base', dataset_container.size)
        self._dataset = dataset.batch('batched', 3)

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_single(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[0].x),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[0:3]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[1].x),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[3:6]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[2].x),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[6:9]], 0)))

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_single_wrapped(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[3].x),
                           ivy.to_numpy(ivy.concatenate(
                               [ivy.expand_dims(a, 0) for a in self._x[-1:] + self._x[0:2]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[4].x),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[2:5]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[-1].x),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[7:10]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[-2].x),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[4:7]], 0)))

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_slice(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[0:2].x[0]),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[0:3]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[0:2].x[1]),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[3:6]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[2:3].x[0]),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[6:9]], 0)))

    @pytest.mark.parametrize(
        "array_shape", [[1], []])
    def test_slice_wrapped(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[0]),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[7:10]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[1]),
                           ivy.to_numpy(ivy.concatenate([ivy.expand_dims(a, 0) for a in self._x[0:3]], 0)))
        assert np.allclose(ivy.to_numpy(self._dataset[3:4].x[0]),
                           ivy.to_numpy(ivy.concatenate(
                               [ivy.expand_dims(a, 0) for a in self._x[-1:] + self._x[0:2]], 0)))


class TestUnbatch:

    def _init(self, array_shape):
        x = [ivy.array([[0], [1], [2]]), ivy.array([[3], [4], [5]]), ivy.array([[6], [7], [8]])]
        self._x = [ivy.reshape(item, array_shape) for item in x]
        dataset_container = ivy.Container({'x': x})
        dataset = Dataset(dataset_container, 'base', dataset_container.size)
        self._dataset = dataset.unbatch('unbatched', 9)

    @pytest.mark.parametrize(
        "array_shape", [[3, 1], [3]])
    def test_single(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[0].x), ivy.to_numpy(self._x[0][0]))
        assert np.allclose(ivy.to_numpy(self._dataset[1].x), ivy.to_numpy(self._x[0][1]))
        assert np.allclose(ivy.to_numpy(self._dataset[2].x), ivy.to_numpy(self._x[0][2]))
        assert np.allclose(ivy.to_numpy(self._dataset[8].x), ivy.to_numpy(self._x[2][2]))

    @pytest.mark.parametrize(
        "array_shape", [[3, 1], [3]])
    def test_single_wrapped(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[9].x), ivy.to_numpy(self._x[0][0]))
        assert np.allclose(ivy.to_numpy(self._dataset[-1].x), ivy.to_numpy(self._x[2][2]))
        assert np.allclose(ivy.to_numpy(self._dataset[-2].x), ivy.to_numpy(self._x[2][1]))

    @pytest.mark.parametrize(
        "array_shape", [[3, 1], [3]])
    def test_slice(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[0:2].x[0]), ivy.to_numpy(self._x[0][0]))
        assert np.allclose(ivy.to_numpy(self._dataset[0:2].x[1]), ivy.to_numpy(self._x[0][1]))
        assert np.allclose(ivy.to_numpy(self._dataset[6:8].x[0]), ivy.to_numpy(self._x[2][0]))
        assert np.allclose(ivy.to_numpy(self._dataset[6:8].x[1]), ivy.to_numpy(self._x[2][1]))

    @pytest.mark.parametrize(
        "array_shape", [[3, 1], [3]])
    def test_slice_wrapped(self, dev_str, f, call, array_shape):

        self._init(array_shape)

        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[0]), ivy.to_numpy(self._x[2][2]))
        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[1]), ivy.to_numpy(self._x[0][0]))
        assert np.allclose(ivy.to_numpy(self._dataset[8:10].x[0]), ivy.to_numpy(self._x[2][2]))
        assert np.allclose(ivy.to_numpy(self._dataset[8:10].x[1]), ivy.to_numpy(self._x[0][0]))


class TestUnbatchAndBatch:

    def _init(self):
        self._x = [ivy.array([0, 1]),
                   ivy.array([2, 3, 4, 5, 6, 7, 8, 9])]
        dataset_container = ivy.Container({'x': self._x})
        dataset = Dataset(dataset_container, 'base', dataset_container.size)
        dataset = dataset.unbatch('unbatched', 10)
        self._dataset = dataset.batch('batched', 3)

    def test_single(self, dev_str, f, call):

        self._init()

        assert np.allclose(ivy.to_numpy(self._dataset[0].x), ivy.to_numpy(ivy.array([0, 1, 2])))
        assert np.allclose(ivy.to_numpy(self._dataset[1].x), ivy.to_numpy(ivy.array([3, 4, 5])))
        assert np.allclose(ivy.to_numpy(self._dataset[2].x), ivy.to_numpy(ivy.array([6, 7, 8])))
        assert np.allclose(ivy.to_numpy(self._dataset[3].x), ivy.to_numpy(ivy.array([9, 0, 1])))

    def test_single_wrapped(self, dev_str, f, call):

        self._init()

        assert np.allclose(ivy.to_numpy(self._dataset[3].x), ivy.to_numpy(ivy.array([9, 0, 1])))
        assert np.allclose(ivy.to_numpy(self._dataset[4].x), ivy.to_numpy(ivy.array([2, 3, 4])))
        assert np.allclose(ivy.to_numpy(self._dataset[5].x), ivy.to_numpy(ivy.array([5, 6, 7])))
        assert np.allclose(ivy.to_numpy(self._dataset[6].x), ivy.to_numpy(ivy.array([8, 9, 0])))
        assert np.allclose(ivy.to_numpy(self._dataset[7].x), ivy.to_numpy(ivy.array([1, 2, 3])))
        assert np.allclose(ivy.to_numpy(self._dataset[8].x), ivy.to_numpy(ivy.array([4, 5, 6])))
        assert np.allclose(ivy.to_numpy(self._dataset[9].x), ivy.to_numpy(ivy.array([7, 8, 9])))

        assert np.allclose(ivy.to_numpy(self._dataset[-1].x), ivy.to_numpy(ivy.array([7, 8, 9])))
        assert np.allclose(ivy.to_numpy(self._dataset[-2].x), ivy.to_numpy(ivy.array([4, 5, 6])))
        assert np.allclose(ivy.to_numpy(self._dataset[-3].x), ivy.to_numpy(ivy.array([1, 2, 3])))
        assert np.allclose(ivy.to_numpy(self._dataset[-4].x), ivy.to_numpy(ivy.array([8, 9, 0])))
        assert np.allclose(ivy.to_numpy(self._dataset[-5].x), ivy.to_numpy(ivy.array([5, 6, 7])))

    def test_slice(self, dev_str, f, call):

        self._init()

        assert np.allclose(ivy.to_numpy(self._dataset[0:2].x[0]), ivy.to_numpy(ivy.array([0, 1, 2])))
        assert np.allclose(ivy.to_numpy(self._dataset[0:2].x[1]), ivy.to_numpy(ivy.array([3, 4, 5])))
        assert np.allclose(ivy.to_numpy(self._dataset[2:3].x[0]), ivy.to_numpy(ivy.array([6, 7, 8])))

    def test_slice_wrapped(self, dev_str, f, call):

        self._init()

        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[0]), ivy.to_numpy(ivy.array([7, 8, 9])))
        assert np.allclose(ivy.to_numpy(self._dataset[-1:1].x[1]), ivy.to_numpy(ivy.array([0, 1, 2])))
        assert np.allclose(ivy.to_numpy(self._dataset[2:4].x[0]), ivy.to_numpy(ivy.array([6, 7, 8])))
        assert np.allclose(ivy.to_numpy(self._dataset[2:4].x[1]), ivy.to_numpy(ivy.array([9, 0, 1])))
