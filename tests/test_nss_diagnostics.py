"""Count actual slice evaluations and time completed NSS transitions."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from blackjax.ns.base import init_state_strategy
from blackjax.ns.nss import covariance_proposal, slice_constrained_step
from blackjax.mcmc.slice import build_kernel, stepping_out
from ceridwen.sampler.nested import (BlackJAXNestedSamplerAdapter,
                                     stepping_out_carry)


def test_logical_call_count_includes_endpoint_checks_and_shrinkage():
    calls = []

    def loglike(position):
        calls.append(1)
        return -jnp.sum(position['x'] ** 2)

    def initialize(position, loglikelihood_birth=jnp.nan):
        return init_state_strategy(position, lambda p: jnp.zeros(()),
                                   loglike, loglikelihood_birth)

    move = slice_constrained_step(
        initialize, build_kernel(interval=stepping_out,
                                 max_expansions=10, max_shrinkage=100),
        covariance_proposal)
    with jax.disable_jit():
        state = initialize({'x': jnp.array([0.1, 0.2])})
        calls.clear()
        _, info = move(jax.random.PRNGKey(37), state, -2., cov=jnp.eye(2))
        counted = BlackJAXNestedSamplerAdapter._logical_likelihood_calls(
            SimpleNamespace(update_info=info))
    assert int(counted) == len(calls)
    assert len(calls) > 1


def test_logical_call_count_sums_particles_and_moves():
    info = SimpleNamespace(update_info=SimpleNamespace(
        num_expansions=jnp.array([[0, 1], [2, 3]]),
        num_shrink=jnp.array([[1, 4], [2, 1]])))
    assert int(BlackJAXNestedSamplerAdapter._logical_likelihood_calls(info)) == 22


def test_likelihood_count_semantics_survive_hdf5_and_legacy_loading(tmp_path):
    import h5py
    from ceridwen.fit import write_result_h5, load_result_h5
    from ceridwen.sampler.runner import SamplingResult

    model = SimpleNamespace(observations=[], csp=SimpleNamespace(wave=jnp.arange(2.)),
                            theta_init={'x': jnp.zeros(1)}, priors={}, transforms={},
                            param_names=['x'], zred=0.)
    result = SamplingResult(samples={'x': jnp.zeros(2)}, log_evidence=0.,
        log_evidence_err=.1, log_weights=jnp.zeros(2), log_likelihoods=jnp.zeros(2),
        param_names=['x'], n_likelihood_calls=13, wall_time_s=1., sampler_name='blackjax.nss',
        likelihood_count_kind='logical_including_initialization')
    path = tmp_path/'result.h5'
    write_result_h5(path, model, result)
    loaded = load_result_h5(path)
    assert loaded.n_likelihood_calls == 13
    assert loaded.likelihood_count_kind == 'logical_including_initialization'
    with h5py.File(path, 'r+') as f:
        del f['samples'].attrs['likelihood_count_kind']
    assert load_result_h5(path).likelihood_count_kind == 'legacy_estimate'


def test_callback_replays_completed_transition_and_counts_initial_points():
    from ceridwen.sampler import Uniform
    import numpy as np

    records = []
    adapter = BlackJAXNestedSamplerAdapter(
        {'x': Uniform(low=-5., high=5.)}, num_live=20, num_delete=5,
        num_inner_steps=3, logZ_tol=1., verbose=False,
        iteration_callback=lambda *args: records.append(args))
    result = adapter.run(lambda p: -.5*jnp.sum(p['x']**2),
                         lambda p: jnp.zeros(()), {'x': jnp.zeros(2)},
                         jax.random.PRNGKey(123))
    assert records
    count = 20
    for number, key, incoming, outgoing, info, step, elapsed in records:
        assert number > 0 and elapsed > 0
        replay = step(key, incoming)
        for a, b in zip(jax.tree.leaves((outgoing, info)), jax.tree.leaves(replay)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        count += int(adapter._logical_likelihood_calls(info))
    assert result.n_likelihood_calls == count


def test_carry_stepping_out_counts_the_same_calls_as_stock():
    counts = {}
    for interval in (stepping_out, stepping_out_carry):
        calls = []

        def loglike(position):
            calls.append(1)
            return -jnp.sum(position['x'] ** 2)

        def initialize(position, loglikelihood_birth=jnp.nan):
            return init_state_strategy(position, lambda p: jnp.zeros(()),
                                       loglike, loglikelihood_birth)

        move = slice_constrained_step(
            initialize, build_kernel(interval=interval,
                                     max_expansions=10, max_shrinkage=100),
            covariance_proposal)
        with jax.disable_jit():
            state = initialize({'x': jnp.array([0.1, 0.2])})
            calls.clear()
            move(jax.random.PRNGKey(37), state, -2., cov=jnp.eye(2))
        counts[interval] = len(calls)
    assert counts[stepping_out_carry] == counts[stepping_out] > 1


def test_carry_kernel_is_bitwise_equal_to_stock_blackjax_nss():
    from ceridwen.sampler import Uniform
    import numpy as np

    def run(slice_kernel):
        records = []
        adapter = BlackJAXNestedSamplerAdapter(
            {'x': Uniform(low=-5., high=5.)}, num_live=40, num_delete=8,
            num_inner_steps=6, verbose=False, slice_kernel=slice_kernel,
            iteration_callback=lambda *args: records.append(args[3:5]))
        # A narrow, correlated peak: the stepping-out loops reach their
        # caps early on and the shrink loop runs many times later.
        result = adapter.run(
            lambda p: -50.*jnp.sum((p['x'] - .7)**2) - 40.*jnp.prod(p['x']),
            lambda p: jnp.zeros(()), {'x': jnp.zeros(3)},
            jax.random.PRNGKey(20260930))
        return result, records

    carry, carry_records = run('carry')
    stock, stock_records = run('stock')
    assert len(carry_records) == len(stock_records) > 20
    expansions = np.concatenate([np.asarray(info.update_info.num_expansions).ravel()
                                 for _, info in stock_records])
    assert expansions.min() == 0 and expansions.max() == 9
    for a, b in zip(jax.tree.leaves(carry_records), jax.tree.leaves(stock_records)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for name in ('log_likelihoods', 'log_weights'):
        np.testing.assert_array_equal(np.asarray(getattr(carry, name)),
                                      np.asarray(getattr(stock, name)))
    np.testing.assert_array_equal(np.asarray(carry.samples['x']),
                                  np.asarray(stock.samples['x']))
    assert carry.log_evidence == stock.log_evidence
    assert carry.n_likelihood_calls == stock.n_likelihood_calls
