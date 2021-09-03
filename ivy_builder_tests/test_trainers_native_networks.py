# global
import ivy
import copy
import torch
import pytest
import numpy as np

# local
import ivy_tests.helpers as helpers
from ivy_builder.abstract.network import Network as BaseNetwork


# noinspection PyUnresolvedReferences
class TorchMLP(torch.nn.Module):

    def __init__(self, kernels):
        super(TorchMLP, self).__init__()
        self._dense0 = torch.nn.Linear(kernels, 1, bias=False)

    def forward(self, x):
        return self._dense0(x)


# noinspection PyAttributeOutsideInit
class Network(BaseNetwork):

    def __init__(self, spec):
        super().__init__(spec, v=ivy.Container())

    def _build(self):
        self._mlp = TorchMLP(1)
        self._v = ivy.Container(dict([(str(i), p) for i, p in enumerate(self._mlp.parameters())]))

    def _forward(self, x):
        return self._mlp(x)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        if not value:
            return
        for i, p in enumerate(self._mlp.parameters()):
            p.data = value[str(i)].detach().data


# Tests #
# ------#

def test_trainer_with_native_network(dev_str, call):
    if call is not helpers.torch_call:
        # currently only pytorch network class implemented
        pytest.skip()

    # duplicate networks
    ivy.random.seed(0)
    torch_network = Network(ivy.Container({'device': 'cpu:0'}))
    torch_network.build()
    ivy_network = copy.deepcopy(torch_network)

    # input
    x = torch.ones((1, 1))

    # ivy training
    ivy_optimizer = ivy.optimizers.SGD(lr=0.1)

    # native training
    torch_optimizer = torch.optim.SGD(torch_network._mlp.parameters(), lr=0.1)

    for step in range(3):

        print('\nstep {}\n'.format(step))

        # ivy step
        print('\nivy\n')
        ivy_var_before = ivy.to_numpy(ivy_network.v['0'][0, 0]).item()
        print('var before: {}'.format(ivy_var_before))
        total_loss, grads = ivy.execute_with_gradients(lambda v: torch.mean(ivy_network(x, v=v)), ivy_network.v)
        ivy_loss = ivy.to_numpy(total_loss).item()
        print('loss: {}'.format(ivy_loss))
        ivy_grad = ivy.to_numpy(grads['0'][0, 0]).item()
        print('grad: {}'.format(ivy_grad))
        ivy_network.v = ivy_optimizer.step(ivy_network.v, grads)
        ivy_var_after = ivy.to_numpy(ivy_network.v['0'][0, 0]).item()
        print('var after: {}'.format(ivy_var_after))

        # torch step
        print('\ntorch\n')
        torch_optimizer.zero_grad()
        ret = torch_network(x)
        total_loss = torch.mean(ret)
        total_loss.backward()
        torch_var_before = ivy.to_numpy(torch_network.v['0'][0, 0]).item()
        print('var before: {}'.format(torch_var_before))
        torch_loss = ivy.to_numpy(total_loss).item()
        print('loss: {}'.format(torch_loss))
        torch_grad = ivy.to_numpy(torch_network.v['0'].grad[0, 0])
        print('grad: {}'.format(torch_grad))
        torch_optimizer.step()
        torch_var_after = ivy.to_numpy(torch_network.v['0'][0, 0]).item()
        print('var after: {}'.format(torch_var_after))

        # assertion
        try:
            assert np.allclose(torch_var_before, ivy_var_before)
        except AssertionError:
            raise AssertionError('Assertion failed for step {}, natvie var = {}, ivy var = {}'.format(
                step, torch_var_before, ivy_var_before))
        try:
            assert np.allclose(torch_loss, ivy_loss)
        except AssertionError:
            raise AssertionError('Assertion failed for step {}, natvie loss = {}, ivy loss = {}'.format(
                step, torch_loss, ivy_loss))
        try:
            assert np.allclose(torch_var_after, ivy_var_after)
        except AssertionError:
            raise AssertionError('Assertion failed for step {}, natvie var = {}, ivy var = {}'.format(
                step, torch_var_after, ivy_var_after))
