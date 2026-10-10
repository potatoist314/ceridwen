"""Periodic checkpointing + rescue pickle for the nested-sampling adapter.

BlackJAX-ns has no native checkpoint/resume, so ``BlackJAXNestedSamplerAdapter``
adds:
  * a periodic checkpoint (default every 20 min) that finalises the dead points
    against the live ensemble and dumps a snapshot, so a run killed by the
    scheduler wall-time / a node failure still yields a recoverable posterior;
  * an end-of-run rescue pickle in the same format, so a crash anywhere on the
    post-convergence save path can't discard a multi-hour run (the regression
    that lost job 229509 and the 2026-06-19 FMR / N80-extended runs);
  * ``load_checkpoint`` to read either back;
  * ``resume_from=<periodic checkpoint>``: the checkpoint also carries the raw sampler state
    (live state, dead list, rng key, iteration), and a run started from it continues where the
    killed run stopped and reproduces the uninterrupted run bit for bit.

These tests use a fast 3-D Gaussian toy so they run in well under a second.
"""
import glob
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

# Nested sampling requires the handley-lab blackjax fork (provides blackjax.ns);
# skip cleanly on a stock PyPI blackjax that lacks it.
pytest.importorskip("blackjax.ns")

from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter
from ceridwen.sampler import Uniform


def _toy():
    priors = {k: Uniform(low=-5.0, high=5.0) for k in ("x", "y", "z")}

    def loglike(t):
        v = jnp.array([t["x"][0], t["y"][0], t["z"][0]])
        return -0.5 * jnp.sum(v ** 2)

    def logprior(t):
        return jnp.array(0.0)

    theta_init = {k: jnp.array([0.0]) for k in ("x", "y", "z")}
    return priors, loglike, logprior, theta_init


def test_periodic_checkpoint_and_rescue(tmp_path):
    priors, loglike, logprior, theta_init = _toy()
    # tiny interval -> a partial checkpoint fires on (effectively) every iter.
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False,
        checkpoint_interval_s=1e-6, checkpoint_dir=str(tmp_path))
    res = ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    assert np.isfinite(float(res.log_evidence))

    ckpts = glob.glob(os.path.join(tmp_path, "ns_checkpoint_[0-9]*.pkl"))
    rescues = glob.glob(os.path.join(tmp_path, "ns_raw_dead_*.pkl"))
    assert ckpts, "no periodic checkpoint written"
    assert rescues, "no end-of-run rescue pickle written"

    # Partial checkpoint is a complete, loadable snapshot.
    d = BlackJAXNestedSamplerAdapter.load_checkpoint(ckpts[0])
    assert d["schema_version"] == 2
    assert d["partial"] is True
    assert d["n_dead"] > 0
    assert set(d["positions"]) == {"x", "y", "z"}
    assert np.isfinite(d["logZ"])
    assert d["progress"]["iteration"] > 0
    assert d["progress"]["n_likelihood_calls"] > 0
    frames = glob.glob(os.path.join(tmp_path, "ns_checkpoint_frame_*.pkl"))
    assert frames
    frame = BlackJAXNestedSamplerAdapter.load_checkpoint(frames[0])
    assert frame["n_draws"] <= 128
    assert np.isfinite(frame["ess"])
    # Rescue is the same format, flagged converged.
    assert BlackJAXNestedSamplerAdapter.load_checkpoint(rescues[0])["partial"] is False


def test_checkpoint_recovers_a_posterior(tmp_path):
    """A killed run's checkpoint must reconstruct an anesthetic posterior."""
    anesthetic = __import__("anesthetic")
    priors, loglike, logprior, theta_init = _toy()
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False,
        checkpoint_interval_s=1e-6, checkpoint_dir=str(tmp_path))
    ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))

    ck = BlackJAXNestedSamplerAdapter.load_checkpoint(
        glob.glob(os.path.join(tmp_path, "ns_checkpoint_[0-9]*.pkl"))[0])
    data = np.column_stack([ck["positions"][k].reshape(ck["n_dead"], -1)
                            for k in ("x", "y", "z")])
    ns = anesthetic.NestedSamples(
        data=data, logL=ck["loglikelihood"],
        logL_birth=ck["loglikelihood_birth"], columns=["x", "y", "z"])
    assert np.isfinite(float(ns.logZ()))


def test_checkpoint_disabled_when_no_dir(tmp_path, monkeypatch):
    """No checkpoint dir resolvable -> silently no files (no surprise writes)."""
    monkeypatch.delenv("CERIDWEN_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("CERIDWEN_RESCUE_DIR", raising=False)
    priors, loglike, logprior, theta_init = _toy()
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False,
        checkpoint_interval_s=1e-6, checkpoint_dir=None)
    ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    assert ad._resolve_ckpt_dir() is None
    assert not glob.glob(os.path.join(tmp_path, "*.pkl"))


def test_progress_file_records_every_iteration(tmp_path):
    """One JSON line per iteration, with rates, even when verbose is off."""
    import json

    priors, loglike, logprior, theta_init = _toy()
    path = tmp_path / "ns_progress.jsonl"
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False, checkpoint_dir=str(tmp_path),
        progress_path=str(path))
    res = ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["iteration"] for r in records] == list(range(1, len(records) + 1))
    assert records[-1]["n_discarded"] == 40 * len(records)
    assert records[-1]["n_likelihood_calls"] == res.n_likelihood_calls
    calls = [r["n_likelihood_calls"] for r in records]
    assert calls == sorted(calls)
    for r in records:
        assert r["iteration_s"] > 0
        assert np.isclose(r["dead_per_s"], r["n_discarded"] / r["elapsed_s"])
        assert np.isclose(r["likelihood_calls_per_s"],
                          r["n_likelihood_calls"] / r["elapsed_s"])
        assert np.isclose(r["delta_logZ"], r["logZ_live"] - r["logZ"])
    # The rescue snapshot reports the same final iteration.
    rescue = glob.glob(os.path.join(tmp_path, "ns_raw_dead_*.pkl"))[0]
    progress = BlackJAXNestedSamplerAdapter.load_checkpoint(rescue)["progress"]
    assert progress["iteration"] == records[-1]["iteration"]


def test_no_progress_file_by_default(tmp_path):
    priors, loglike, logprior, theta_init = _toy()
    ad = BlackJAXNestedSamplerAdapter(
        priors, num_live=120, num_delete=40, num_inner_steps=10,
        logZ_tol=-2.0, verbose=False, checkpoint_dir=str(tmp_path))
    ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    assert not glob.glob(os.path.join(tmp_path, "*.jsonl"))


def test_load_checkpoint_accepts_legacy_snapshot(tmp_path):
    import pickle

    path = tmp_path / "legacy.pkl"
    legacy = {
        "positions": {"x": np.zeros((2, 1))},
        "loglikelihood": np.zeros(2),
        "loglikelihood_birth": np.full(2, -1.0),
        "logZ": 0.0,
        "n_dead": 2,
        "partial": True,
    }
    with path.open("wb") as handle:
        pickle.dump(legacy, handle)

    loaded = BlackJAXNestedSamplerAdapter.load_checkpoint(path)
    assert loaded.keys() == legacy.keys()
    assert loaded["partial"] is True
    assert loaded["n_dead"] == 2
    np.testing.assert_array_equal(loaded["positions"]["x"], legacy["positions"]["x"])
class _Killed(Exception):
    pass


def _run_killed_after(monkeypatch, n_iter, **kw):
    """Run the toy and raise inside iteration ``n_iter`` (before its checkpoint is written),
    as a scheduler kill would; the checkpoint on disk is then that of iteration n_iter - 1."""
    import tqdm
    real = tqdm.tqdm

    class _Bar(real):
        calls = 0

        def update(self, n=1):
            type(self).calls += 1
            if type(self).calls >= n_iter:
                raise _Killed
            return super().update(n)

    monkeypatch.setattr(tqdm, "tqdm", _Bar)
    priors, loglike, logprior, theta_init = _toy()
    ad = BlackJAXNestedSamplerAdapter(priors, **kw)
    with pytest.raises(_Killed):
        ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    monkeypatch.setattr(tqdm, "tqdm", real)


_KW = dict(num_live=120, num_delete=40, num_inner_steps=10, logZ_tol=-2.0, verbose=False,
           checkpoint_interval_s=1e-6)


def _equal(a, b):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_resume_reproduces_the_uninterrupted_run(tmp_path, monkeypatch):
    priors, loglike, logprior, theta_init = _toy()
    full = BlackJAXNestedSamplerAdapter(priors, checkpoint_dir=str(tmp_path / "full"), **_KW)
    ref = full.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    n_iter_total = ref.log_likelihoods.shape[0] // 40 - 3   # dead batches before finalise
    assert n_iter_total > 4

    _run_killed_after(monkeypatch, 4, checkpoint_dir=str(tmp_path / "killed"), **_KW)
    ck_path = glob.glob(os.path.join(tmp_path / "killed", "ns_checkpoint_[0-9]*.pkl"))[0]
    ck = BlackJAXNestedSamplerAdapter.load_checkpoint(ck_path)
    assert ck["resume"]["n_iter"] == 3 and ck["partial"] is True

    ad = BlackJAXNestedSamplerAdapter(priors, checkpoint_dir=str(tmp_path / "resumed"),
                                      resume_from=ck_path, **_KW)
    res = ad.run(loglike, logprior, theta_init, jax.random.PRNGKey(0))
    for k in theta_init:
        assert _equal(res.samples[k], ref.samples[k]), k
    assert _equal(res.log_likelihoods, ref.log_likelihoods)
    assert _equal(res.log_likelihoods_birth, ref.log_likelihoods_birth)
    assert _equal(res.log_weights, ref.log_weights)
    assert res.log_evidence == ref.log_evidence
    assert res.n_likelihood_calls == ref.n_likelihood_calls


def test_resume_refuses_a_foreign_checkpoint(tmp_path, monkeypatch):
    _run_killed_after(monkeypatch, 3, checkpoint_dir=str(tmp_path), **_KW)
    ck_path = glob.glob(os.path.join(tmp_path, "ns_checkpoint_[0-9]*.pkl"))[0]
    priors, loglike, logprior, theta_init = _toy()
    key = jax.random.PRNGKey(0)

    other = dict(_KW, num_delete=20)
    with pytest.raises(ValueError, match="num_delete"):
        BlackJAXNestedSamplerAdapter(priors, resume_from=ck_path, **other).run(
            loglike, logprior, theta_init, key)
    with pytest.raises(ValueError, match="params"):
        BlackJAXNestedSamplerAdapter(priors, resume_from=ck_path, **_KW).run(
            loglike, logprior, {**theta_init, "z": jnp.zeros(2)}, key)
    with pytest.raises(ValueError, match="another model or data set"):
        BlackJAXNestedSamplerAdapter(priors, resume_from=ck_path, **_KW).run(
            lambda t: 2.0 * loglike(t), logprior, theta_init, key)
    with pytest.raises(FileNotFoundError):
        BlackJAXNestedSamplerAdapter(priors, resume_from=str(tmp_path / "nope.pkl"))

    # Q1-006: the same model re-evaluated at another batch width moves ln L in the last
    # float32-contraction digits (1.4e-8 relative on the README model); that is not a
    # foreign checkpoint and must resume
    res = BlackJAXNestedSamplerAdapter(priors, resume_from=ck_path, **_KW).run(
        lambda t: loglike(t) * (1.0 + 1e-8), logprior, theta_init, key)
    assert np.isfinite(float(res.log_evidence))

    # a checkpoint without the sampler state (older format, or a rescue pickle)
    import pickle
    ck = BlackJAXNestedSamplerAdapter.load_checkpoint(ck_path)
    ck.pop("resume")
    old = tmp_path / "old.pkl"
    old.write_bytes(pickle.dumps(ck))
    with pytest.raises(ValueError, match="holds no sampler state"):
        BlackJAXNestedSamplerAdapter(priors, resume_from=str(old), **_KW).run(
            loglike, logprior, theta_init, key)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        test_periodic_checkpoint_and_rescue(Path(d))
    print("checkpoint tests passed")
