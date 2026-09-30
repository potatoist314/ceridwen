"""
ceridwen/sampler/nested.py
==========================
BlackJAX adaptive nested sampler (``blackjax.nss``) adapter for Ceridwen.

Algorithm background
--------------------
Nested sampling (Skilling 2006) integrates the Bayesian evidence by
compressing the prior volume along iso-likelihood contours.  At each
step the least-likely live point is replaced by a new point drawn from
the prior inside the current likelihood contour.

BlackJAX's ``nss`` (nested slice sampler) uses an adaptive slice-sampling
inner kernel.  The key parameters are:

``num_live``
    Resolution parameter.  Evidence uncertainty scales as
    :math:`1/\\sqrt{N_\\mathrm{live}}`.  Typical: 200 (fast) – 2000
    (publication quality).

``num_inner_steps``
    Reliability of the inner MCMC chain.  Best practice: ``n_dims * 5``.
    Check stability by halving/doubling.

``num_delete``
    Parallelisation: points removed per iteration.
    Default: ``max(1, num_live // 5)``, adopted from the 2026-08 JADES
    campaign tuning (64 at ``num_live = 300``).  Larger batches gain
    little extra throughput once evaluation is vectorised, while the
    naive single-deletion volume rule used by fallback evidence
    estimators (not the anesthetic path) biases :math:`\\ln Z` high by
    several nats at ``num_delete = num_live // 2``.  Must stay
    strictly below ``num_live`` (a BlackJAX NSS constraint).

``logZ_tol``
    Convergence threshold :math:`\\ln(Z_\\mathrm{live}/Z)`.
    Iteration stops when the live contribution drops below this.
    Default ``-5``.  The fraction of the total evidence still in the
    live set at termination is :math:`f = e^{tol}/(1 + e^{tol})`:
    ``-5`` leaves 0.67 per cent (marginally stricter than nautilus's
    ``f_live = 0.01``, which corresponds to ``-4.595``); ``-3`` leaves
    4.7 per cent; a value as loose as ``-1`` leaves 27 per cent of the
    evidence unaccounted for.

Installation
------------
::

    # Nested sampling (blackjax.nss) is merged into the official blackjax;
    # until it lands in a PyPI release, install the pinned commit:
    pip install "git+https://github.com/blackjax-devs/blackjax@f73e12956"
    pip install anesthetic   # optional (posteriors/plots; now on PyPI)

Usage
-----
::

    from ceridwen.sampler        import run_sampler
    from ceridwen.sampler.nested import BlackJAXNestedSamplerAdapter

    adapter = BlackJAXNestedSamplerAdapter(
        priors          = model.priors,
        num_live        = 500,
        num_inner_steps = len(model.param_names) * 5,
    )
    result = run_sampler(model, multi_likelihood, adapter,
                         jax.random.PRNGKey(42))

    print(result.summary())
    ns = result.to_anesthetic(labels={...})
    ns.plot_2d([...])
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .runner import SamplerAdapter, SamplingResult

Array = jax.Array


def stepping_out_carry(rng_key, in_slice, width, max_expansions):
    """Neal (2003) Fig. 3 stepping-out with the slice test in the loop body.

    Drop-in for ``blackjax.mcmc.slice.stepping_out``: same random numbers,
    same sequence of ``in_slice`` evaluations, bitwise-equal bracket.  The
    stock loops call ``in_slice`` in their condition, and a vmapped
    ``lax.while_loop`` evaluates the condition a second time inside its
    body, so every expansion costs two batched likelihood evaluations.
    Here the body evaluates the new edge once and carries the result.
    """
    u_key, jk_key = jax.random.split(rng_key)
    u = jax.random.uniform(u_key)
    left = -width * u
    right = left + width
    v = jax.random.uniform(jk_key)
    j = jnp.floor(max_expansions * v).astype(int)
    k = (max_expansions - 1) - j

    def cond(carry):
        _, n, inside = carry
        return inside & (n > 0)

    def step(sign):
        def body(carry):
            edge, n, _ = carry
            edge = edge + sign * width
            return edge, n - 1, in_slice(edge)
        return body

    left, jl, _ = jax.lax.while_loop(cond, step(-1), (left, j, in_slice(left)))
    right, kr, _ = jax.lax.while_loop(cond, step(+1), (right, k, in_slice(right)))
    return left, right, (j - jl) + (k - kr), lambda t: jnp.asarray(True)


_LEFT, _RIGHT, _SHRINK = 0, 1, 2


def lane_update(logprior_fn, loglikelihood_fn, num_inner_steps, num_delete,
                max_expansions=10, max_shrinkage=100, width=1.0, lookahead=4):
    """Inner update of the NS kernel: every particle's slice chain as one lane.

    ``blackjax.nss`` replaces ``num_delete`` particles per iteration.  Each
    particle (a lane) runs ``num_inner_steps`` slice steps, and each slice
    step runs three ``lax.while_loop``s: two stepping-out loops and one
    shrink loop.  Under ``vmap`` a while loop runs until its slowest lane
    stops, so at every slice step every lane pays for the longest loop of
    any lane.

    Here the lanes share one loop.  Each round evaluates one batch of
    ``num_delete`` candidates, one per lane; the lane's phase (left edge,
    right edge, shrink) selects its candidate, and a lane that completes a
    slice step starts its next step in the next round.  This is the
    finite-state-machine vectorisation of Dance et al. (2025,
    arXiv:2503.17405).  Two further rules remove rounds:

    * A candidate below the slice level of the prior is outside the slice
      whatever its likelihood, so the lane moves on without a likelihood
      evaluation.
    * Once some lanes have finished, their batch slots evaluate candidates
      that the running lanes need later if their current candidates are
      outside the slice: the next edge, then the shrink proposals (up to
      ``lookahead - 1`` per lane, lanes with the most slice steps left
      first).  A lane takes a stored result when its candidate is the same
      slice step, the same point on the slice line and the same position.

    The random numbers are split as BlackJAX splits them and every
    candidate is evaluated with the same arithmetic in a batch of
    ``num_delete``, so the new particles and the per-step ``SliceInfo``
    are bitwise equal to those of ``blackjax.nss``.

    Drop-in for ``blackjax.ns.from_mcmc.update_with_mcmc_take_last`` applied
    to ``blackjax.ns.nss.slice_constrained_step`` with the stepping-out
    interval and the covariance proposal.  Returns the ``num_delete`` new
    particles and a ``SliceInfo`` of shape ``(num_delete, num_inner_steps)``.
    """
    from functools import partial
    from blackjax.mcmc.slice import SliceInfo
    from blackjax.ns.base import init_state_strategy
    from blackjax.ns.nss import sample_direction_from_covariance

    slots = 2 * lookahead      # stored results per lane

    def update_function(rng_key, state, loglikelihood_0, cov):
        choice_key, sample_key = jax.random.split(rng_key)
        particles = state.particles
        evaluate = jax.vmap(partial(init_state_strategy, logprior_fn=logprior_fn,
                                    loglikelihood_fn=loglikelihood_fn,
                                    loglikelihood_birth=loglikelihood_0))

        # Start particles, selected as in update_with_mcmc_take_last.
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key, len(weights), shape=(num_delete,),
            p=weights / weights.sum(), replace=True)
        start = jax.tree.map(lambda x: x[start_idx], particles)
        _, unravel = ravel_pytree(jax.tree.map(lambda x: x[0], start.position))

        def lane_draws(rng_key, position):
            def draws(key):
                # The random numbers of one slice step: blackjax.mcmc.slice
                # build_kernel, _univariate_slice and stepping_out.
                prop_key, slice_key = jax.random.split(key)
                direction, _ = ravel_pytree(sample_direction_from_covariance(
                    prop_key, position, cov))
                level_key, interval_key, shrink_key = jax.random.split(slice_key, 3)
                log_u = jnp.log(jax.random.uniform(level_key))
                u_key, jk_key = jax.random.split(interval_key)
                left = -width * jax.random.uniform(u_key)
                right = left + width
                j = jnp.floor(max_expansions * jax.random.uniform(jk_key)).astype(int)
                k = (max_expansions - 1) - j
                return direction, jnp.stack([log_u, left, right]), jnp.stack([j, k]), shrink_key

            # A scan, as in BlackJAX, so the draws keep their batch structure.
            return jax.lax.map(draws, jax.random.split(rng_key, num_inner_steps))

        directions, reals, caps, shrink_keys = jax.vmap(lane_draws)(
            jax.random.split(sample_key, num_delete), start.position)
        counter = caps.dtype.type
        lanes = jnp.arange(num_delete)

        # Per-lane functions, vmapped over the lane index l.
        def begin(l, i, logdensity):
            log_u, left, right = reals[l, i]
            return dict(phase=jnp.asarray(_LEFT), level=logdensity + log_u,
                        left=left, right=right,
                        n_left=caps[l, i, 0], n_right=caps[l, i, 1],
                        bracket_left=left, bracket_right=right,
                        key=shrink_keys[l, i], n_expansions=counter(0),
                        n_shrink=counter(0))

        def propose(phase, left, right, key):
            key, subkey = jax.random.split(key)
            t = left + jax.random.uniform(subkey) * (right - left)
            return jnp.where(phase == _LEFT, left, jnp.where(phase == _RIGHT, right, t)), key

        def point(l, i, position, t):
            direction = unravel(directions[l, i])
            return jax.tree.map(lambda p, d: p + t * d, position, direction)

        def look(l, i, lane, s, stored):
            """The lane's candidate, its prior, and a stored result for it."""
            t, key = propose(s["phase"], s["left"], s["right"], s["key"])
            x = point(l, jnp.minimum(i, num_inner_steps - 1), lane.position, t)
            # The stored point must equal this one bitwise: XLA may contract
            # p + t * d into an FMA in one layout and not in another.
            match = (stored["step"][l] == i) & (stored["t"][l] == t) & jnp.all(jnp.stack([
                jnp.all((v[l] == u).reshape(slots, -1), axis=1)
                for v, u in zip(jax.tree.leaves(stored["state"].position), jax.tree.leaves(x))]), axis=0)
            found = jax.tree.map(lambda v: v[l, jnp.argmax(match)], stored["state"])
            return dict(t=t, key=key, x=x, logprior=logprior_fn(x),
                        hit=jnp.any(match), stored=found)

        def advance(l, active, i, lane, s, info, t, key, candidate):
            """One transition of the lane's state machine for an evaluated candidate."""
            in_left, in_right = s["phase"] == _LEFT, s["phase"] == _RIGHT
            in_shrink = s["phase"] == _SHRINK
            inside = ((candidate.logdensity >= s["level"])
                      & (candidate.loglikelihood > loglikelihood_0))

            # Stepping out (Neal 2003, Fig. 3).
            grow_left = in_left & inside & (s["n_left"] > 0)
            grow_right = in_right & inside & (s["n_right"] > 0)
            left = jnp.where(grow_left, s["left"] - width, s["left"])
            right = jnp.where(grow_right, s["right"] + width, s["right"])
            bracket_left = jnp.where(in_shrink, s["bracket_left"], left)
            bracket_right = jnp.where(in_shrink, s["bracket_right"], right)
            n_expansions = s["n_expansions"] + (grow_left | grow_right)
            phase = jnp.where(in_left & ~grow_left, _RIGHT,
                              jnp.where(in_right & ~grow_right, _SHRINK, s["phase"]))

            # Shrinkage (Neal 2003, Fig. 5).
            found = active & in_shrink & inside
            left = jnp.where(in_shrink & (t < 0.0), t, left)
            right = jnp.where(in_shrink & (t >= 0.0), t, right)
            n_shrink = s["n_shrink"] + in_shrink
            lane = jax.tree.map(
                lambda new, old: jnp.where(found, new, old), candidate, lane)
            done = active & in_shrink & (found | (n_shrink >= max_shrinkage))

            row = jnp.stack([found, n_expansions, n_shrink,
                             bracket_left, bracket_right]).astype(info.dtype)
            info = info.at[jnp.where(done, i, num_inner_steps)].set(row, mode="drop")
            running = dict(
                phase=phase, level=s["level"], left=left, right=right,
                n_left=s["n_left"] - grow_left, n_right=s["n_right"] - grow_right,
                bracket_left=bracket_left, bracket_right=bracket_right,
                key=jnp.where(in_shrink, key, s["key"]),
                n_expansions=n_expansions, n_shrink=n_shrink)
            running = jax.tree.map(lambda a, b: jnp.where(active, a, b), running, s)
            i = i + done
            fresh = begin(l, jnp.minimum(i, num_inner_steps - 1), lane.logdensity)
            s = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, running)
            return i, lane, s, info

        def ahead(l, i, lane, s, now):
            """The next ``lookahead - 1`` candidates if the current one and each
            of these is outside the slice, and whether they belong to the
            current slice step."""
            phase, left, right, key = s["phase"], s["left"], s["right"], s["key"]
            n_shrink, t, next_key = s["n_shrink"], now["t"], now["key"]
            same_step, out = jnp.asarray(True), []
            for _ in range(lookahead - 1):
                in_shrink = phase == _SHRINK
                left = jnp.where(in_shrink & (t < 0.0), t, left)
                right = jnp.where(in_shrink & (t >= 0.0), t, right)
                key = jnp.where(in_shrink, next_key, key)
                n_shrink = n_shrink + in_shrink
                same_step = same_step & ~(in_shrink & (n_shrink >= max_shrinkage))
                phase = jnp.where(phase == _LEFT, _RIGHT, jnp.where(phase == _RIGHT, _SHRINK, phase))
                t, next_key = propose(phase, left, right, key)
                x = point(l, jnp.minimum(i, num_inner_steps - 1), lane.position, t)
                out.append((t, x, logprior_fn(x) >= s["level"], same_step))
            return jax.tree.map(lambda *v: jnp.stack(v), *out)

        v_look = jax.vmap(look, in_axes=(0, 0, 0, 0, None))
        v_advance = jax.vmap(advance)
        v_ahead = jax.vmap(ahead)

        def settle(carry):
            """Move every lane past candidates outside the prior slice level or
            with a stored result."""
            def free(carry):
                i, lane, s, info, stored, now = carry
                return (i < num_inner_steps) & (now["hit"] | ~(now["logprior"] >= s["level"]))

            def body(carry):
                i, lane, s, info, stored, now = carry
                active = free(carry)
                # Outside the prior level: logdensity < level makes it outside
                # the slice whatever the likelihood.
                below = lane._replace(position=now["x"], logdensity=now["logprior"])
                candidate = jax.tree.map(
                    lambda a, b: jnp.where(jnp.expand_dims(now["hit"], tuple(range(1, a.ndim))), a, b),
                    now["stored"], below)
                i, lane, s, info = v_advance(lanes, active, i, lane, s, info,
                                             now["t"], now["key"], candidate)
                return i, lane, s, info, stored, v_look(lanes, i, lane, s, stored)

            return jax.lax.while_loop(lambda c: jnp.any(free(c)), body, carry)

        def round_(carry):
            i, lane, s, info, stored, now = carry
            running = i < num_inner_steps
            spare = num_delete - running.sum()

            # Idle slots: the lookahead candidates of the running lanes, by
            # depth, then by slice steps left.
            t_next, x_next, in_prior, same_step = v_ahead(lanes, i, lane, s, now)
            new = ~((stored["step"][:, None, :] == i[:, None, None])
                    & (stored["t"][:, None, :] == t_next[..., None])).any(-1)
            wanted = running[:, None] & same_step & in_prior & new
            depth = jnp.arange(1, lookahead)[None, :]
            score = jnp.where(wanted, depth * (num_inner_steps + 1) + i[:, None],
                              lookahead * (num_inner_steps + 1)).ravel()
            rank = jnp.empty_like(score).at[jnp.argsort(score)].set(
                jnp.arange(score.size)).reshape(wanted.shape)
            chosen = wanted & (rank < spare)
            idle = jnp.argsort(running)          # finished lanes first
            slot = jnp.where(chosen, idle[jnp.minimum(rank, num_delete - 1)], num_delete)

            batch = jax.tree.map(
                lambda a, b: a.at[slot.ravel()].set(
                    b.reshape(-1, *b.shape[2:]), mode="drop"),
                now["x"], x_next)
            evaluated = evaluate(batch)

            # Store every result under its lane, slice step and t.
            take = jnp.concatenate([running[:, None], chosen], axis=1)
            source = jnp.concatenate([lanes[:, None], slot], axis=1)
            where_to = (stored["next"][:, None] + jnp.cumsum(take, axis=1) - 1) % slots
            where_to = jnp.where(take, where_to, slots)
            ts = jnp.concatenate([now["t"][:, None], t_next], axis=1)
            put = lambda old, new: old.at[lanes[:, None], where_to].set(new, mode="drop")  # noqa: E731
            stored = dict(
                step=put(stored["step"], jnp.broadcast_to(i[:, None], take.shape)),
                t=put(stored["t"], ts),
                state=jax.tree.map(
                    lambda old, new: put(old, new[jnp.minimum(source, num_delete - 1)]),
                    stored["state"], evaluated),
                next=stored["next"] + take.sum(1, dtype=stored["next"].dtype))

            i, lane, s, info = v_advance(lanes, running, i, lane, s, info,
                                         now["t"], now["key"], evaluated)
            return settle((i, lane, s, info, stored, v_look(lanes, i, lane, s, stored)))

        i = jnp.zeros(num_delete, dtype=jnp.int32)
        s = jax.vmap(begin)(lanes, i, start.logdensity)
        info = jnp.zeros((num_delete, num_inner_steps, 5), reals.dtype)
        stored = dict(
            step=jnp.full((num_delete, slots), -1, dtype=i.dtype),
            t=jnp.zeros((num_delete, slots), reals.dtype),
            state=jax.tree.map(
                lambda v: jnp.zeros((num_delete, slots, *v.shape[1:]), v.dtype), start),
            next=jnp.zeros(num_delete, dtype=i.dtype))
        carry = settle((i, start, s, info, stored, v_look(lanes, i, start, s, stored)))
        _, lane, _, info, _, _ = jax.lax.while_loop(
            lambda c: jnp.any(c[0] < num_inner_steps), round_, carry)
        return lane, SliceInfo(
            is_accepted=info[..., 0] != 0,
            num_expansions=info[..., 1].astype(caps.dtype),
            num_shrink=info[..., 2].astype(caps.dtype),
            bracket_left=info[..., 3], bracket_right=info[..., 4])

    return update_function


class BlackJAXNestedSamplerAdapter(SamplerAdapter):
    """
    Adapter wrapping ``blackjax.nss`` for use with any Ceridwen ``SedModel``.

    Handles three concerns that are specific to nested sampling in Ceridwen:

    1. **Live-point initialisation** — samples each free parameter
       independently from its registered prior using the ``Prior.sample``
       method from ``ceridwen.sampler.priors``.

    2. **Shape reconciliation** — the BlackJAX NSS live-point dict has
       shape ``{name: (num_live, *param_shape)}``.  The step function
       vmaps over axis 0, delivering single-particle slices of shape
       ``(*param_shape,)`` to ``loglike_fn`` / ``logprior_fn``.  This
       matches the Ceridwen dict-theta convention exactly.

    3. **Evidence extraction** — optionally uses ``anesthetic`` for a
       more accurate :math:`\\ln Z` estimate with uncertainty.

    Parameters
    ----------
    priors : dict[str, Prior]
        Mapping from free-parameter name to a Ceridwen ``Prior`` object.
        **Every free parameter must have a prior** — nested sampling
        requires a proper (normalisable) prior; an improper flat prior
        makes the evidence integral undefined.
    num_live : int, optional
        Number of live points.  Default 500.
    num_inner_steps : int, optional
        Inner MCMC steps per NS iteration.  Default ``n_dims * 5``
        where ``n_dims`` is the total scalar dimension count.
    num_delete : int, optional
        Live points discarded per iteration.  Default
        ``max(1, num_live // 5)`` — see the module docstring for the
        rationale.  Must be strictly below ``num_live``.
    logZ_tol : float, optional
        Convergence threshold on :math:`\\ln(Z_\\mathrm{live}/Z)`.
        Default ``-5.0``, terminating with ~0.7 per cent of the
        evidence still in the live set
        (:math:`f = e^{tol}/(1 + e^{tol})`).
    verbose : bool, optional
        Print a ``tqdm`` progress bar and convergence info.  Default True.
    checkpoint_interval_s : float, optional
        Seconds between periodic checkpoints (default 1200 = 20 min;
        ``<= 0`` disables).  Each checkpoint finalises the accumulated
        dead points against the current live ensemble and dumps a
        snapshot, so a run killed mid-flight still yields a recoverable
        (partial) posterior; the same format is written at convergence
        as the rescue pickle, and :meth:`load_checkpoint` reads either.
    checkpoint_dir : str, optional
        Checkpoint destination.  Resolved at run time as
        ``checkpoint_dir`` → ``$CERIDWEN_CHECKPOINT_DIR`` →
        ``$CERIDWEN_RESCUE_DIR``; when none is set, checkpointing is
        silently skipped (no surprise writes).
    checkpoint_frame_fn : callable, optional
        Function that receives a bounded, equal-weight posterior sample and
        returns a compact serialisable visualization payload.  Periodic and
        final frames are retained separately from the overwritten recovery
        checkpoint, so a later tool can show the actual inference history.
    checkpoint_frame_draws : int, optional
        Maximum equal-weight posterior draws retained per frame.  Default 128.
    checkpoint_frame_limit : int, optional
        Maximum compact frames retained for one process.  Default 64.
    checkpoint_frame_max_bytes : int, optional
        Maximum serialised size of one compact frame.  Default 2 MiB.
    iteration_callback : callable, optional
        Called after each completed step with ``(iteration, key, incoming,
        outgoing, info, compiled_step, elapsed_seconds)``. Default None.
        Intended for profiling; the callback must not modify the sampler state.
    slice_kernel : str, optional
        ``"lanes"`` (default) runs each replaced particle as one loop over
        its whole slice chain (:func:`lane_update`);
        ``"carry"`` runs the BlackJAX slice kernel with
        :func:`stepping_out_carry`; ``"stock"`` runs the unchanged
        ``blackjax.nss`` kernel.  All three give bitwise-equal samples.
    progress_path : str, optional
        JSON-lines file that receives one record per completed iteration:
        the checkpoint ``progress`` fields plus ``iteration_s``,
        ``dead_per_s`` and ``likelihood_calls_per_s`` (both cumulative over
        ``elapsed_s``).  Written whether or not ``verbose`` is set, appended
        and flushed per line so a killed run keeps its history.  Default None
        writes nothing.
    """

    def __init__(
        self,
        priors          : dict,
        num_live        : int   = 500,
        num_inner_steps : Optional[int] = None,
        num_delete      : Optional[int] = None,
        logZ_tol        : float = -5.0,
        verbose         : bool  = True,
        checkpoint_interval_s : float = 1200.0,
        checkpoint_dir        : Optional[str] = None,
        checkpoint_frame_fn   : Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        checkpoint_frame_draws: int = 128,
        checkpoint_frame_limit: int = 64,
        checkpoint_frame_max_bytes: int = 2 * 1024 * 1024,
        iteration_callback: Optional[Callable] = None,
        progress_path: Optional[str] = None,
        slice_kernel: str = "lanes",
    ):
        self.priors          = dict(priors)
        self.num_live        = int(num_live)
        self._num_inner_steps = num_inner_steps   # None → auto
        self._num_delete      = num_delete         # None → max(1, num_live // 5)
        self.logZ_tol        = float(logZ_tol)
        self.verbose         = bool(verbose)
        self.iteration_callback = iteration_callback
        self.progress_path = progress_path
        if slice_kernel not in ("lanes", "carry", "stock"):
            raise ValueError(f"unknown slice_kernel {slice_kernel!r}")
        self.slice_kernel = slice_kernel
        # Periodic checkpointing.  Every ``checkpoint_interval_s`` seconds
        # (default 1200 = 20 min; <= 0 disables) the accumulated dead points
        # are finalised against the current live ensemble and dumped to disk,
        # so a run killed by the scheduler wall-time, a node failure, or any
        # mid-run crash still yields a recoverable (partial) posterior --
        # BlackJAX provides no native checkpointing.  The destination is
        # resolved at run time from ``checkpoint_dir`` ->
        # $CERIDWEN_CHECKPOINT_DIR -> $CERIDWEN_RESCUE_DIR; when none is set
        # checkpointing is silently skipped (no surprise writes).  The same
        # snapshot format is written once more at convergence as the rescue
        # pickle, so :meth:`load_checkpoint` recovers either.
        self.checkpoint_interval_s = float(checkpoint_interval_s)
        self._checkpoint_dir       = checkpoint_dir
        self.checkpoint_frame_fn   = checkpoint_frame_fn
        self.checkpoint_frame_draws = int(checkpoint_frame_draws)
        self.checkpoint_frame_limit = int(checkpoint_frame_limit)
        self.checkpoint_frame_max_bytes = int(checkpoint_frame_max_bytes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_dims(self, theta_init: dict[str, Array]) -> int:
        """Total scalar degrees of freedom."""
        return sum(int(jnp.size(v)) for v in theta_init.values())

    def _sample_prior(
        self,
        theta_init : dict[str, Array],
        rng_key    : Array,
    ) -> dict[str, Array]:
        """
        Draw ``num_live`` initial live points from the registered priors.

        Each parameter is sampled independently.  The returned dict has
        shape ``{name: (num_live, *param_shape)}``.

        Raises
        ------
        ValueError
            If any free parameter has no registered prior.
        """
        particles = {}
        for name, init_val in theta_init.items():
            if name not in self.priors:
                raise ValueError(
                    f"Parameter '{name}' has no prior.  "
                    f"Nested sampling requires a proper prior for every "
                    f"free parameter.  Add '{name}' to the priors dict "
                    f"passed to BlackJAXNestedSamplerAdapter."
                )
            prior          = self.priors[name]
            rng_key, sub   = jax.random.split(rng_key)
            expected_shape = init_val.shape           # e.g. (1,) or (k,)

            # Pass the full target shape (num_live, *expected_shape) to
            # prior.sample so TFP broadcasts a scalar distribution over
            # all required dimensions in a single call.
            #
            # Examples:
            #   Normal(0, 1), expected_shape=(4,)
            #     → sample((500, 4)) → (500, 4)  ✓  iid per element
            #   Uniform(-2.5, 0.2), expected_shape=(1,)
            #     → sample((500, 1)) → (500, 1)  ✓
            #
            # This avoids the shape mismatch that arises when sampling
            # (num_live,) and then trying to reshape to (num_live, k).
            particles[name] = prior.sample(sub, shape=(self.num_live, *expected_shape))

        return particles

    # ------------------------------------------------------------------
    # Checkpoint / rescue helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finalise_dead(live, dead_list, ns_utils):
        """Merge the live ensemble into the accumulated dead points and return
        ``(positions_dict, loglikelihood, loglikelihood_birth)``.

        Version-aware over BlackJAX ``finalise`` (0.1.0b0 has no
        ``update_info`` kwarg) and over the dead-point layout (older dict
        ``particles`` vs v3 ``StateWithLogLikelihood``).  Shared by the
        end-of-run path and the periodic checkpoints so the finalise logic
        lives in ONE place.
        """
        import inspect as _inspect
        if "update_info" in _inspect.signature(ns_utils.finalise).parameters:
            dead = ns_utils.finalise(live, dead_list, update_info=False)
        else:
            dead = ns_utils.finalise(live, dead_list)
        _dp = dead.particles
        if isinstance(_dp, dict):
            return _dp, dead.loglikelihood, dead.loglikelihood_birth
        if hasattr(_dp, "position"):
            return _dp.position, _dp.loglikelihood, _dp.loglikelihood_birth
        positions = dead.position if hasattr(dead, "position") else _dp
        return positions, dead.loglikelihood, dead.loglikelihood_birth

    def _resolve_ckpt_dir(self):
        """Checkpoint destination: explicit arg -> $CERIDWEN_CHECKPOINT_DIR ->
        $CERIDWEN_RESCUE_DIR -> None (disabled)."""
        return (self._checkpoint_dir
                or os.environ.get("CERIDWEN_CHECKPOINT_DIR")
                or os.environ.get("CERIDWEN_RESCUE_DIR"))

    @staticmethod
    def _equal_weight_checkpoint_draws(snapshot, max_draws):
        """Return a deterministic, bounded posterior sample and its ESS."""
        import numpy as _np
        from anesthetic import NestedSamples

        names = list(snapshot["positions"])
        blocks = [
            _np.asarray(snapshot["positions"][name]).reshape(
                snapshot["n_dead"], -1
            )
            for name in names
        ]
        nested = NestedSamples(
            data=_np.hstack(blocks),
            logL=_np.asarray(snapshot["loglikelihood"]),
            logL_birth=_np.asarray(snapshot["loglikelihood_birth"]),
            logzero=float("nan"),
        )
        log_weights = _np.asarray(nested.logw(), dtype=float)
        weights = _np.exp(log_weights - _np.max(log_weights))
        weights /= weights.sum()
        count = min(int(max_draws), snapshot["n_dead"])
        targets = (_np.arange(count, dtype=float) + 0.5) / count
        indices = _np.searchsorted(_np.cumsum(weights), targets, side="left")
        draws = {
            name: _np.asarray(values)[indices]
            for name, values in snapshot["positions"].items()
        }
        ess = float(1.0 / _np.sum(weights**2))
        return draws, ess

    def _dump_checkpoint_frame(self, ckpt_dir, snapshot, progress, *, partial):
        """Write one compact, size-limited posterior frame for visualization."""
        import glob as _glob
        import pickle as _pickle

        if self.checkpoint_frame_draws <= 0 or self.checkpoint_frame_limit <= 0:
            return None
        draws, ess = self._equal_weight_checkpoint_draws(
            snapshot, self.checkpoint_frame_draws
        )
        frame = {
            "schema_version": 1,
            "partial": bool(partial),
            "positions": draws,
            "n_draws": len(next(iter(draws.values()))),
            "ess": ess,
            "progress": dict(progress),
        }
        if self.checkpoint_frame_fn is not None:
            frame["spectrum"] = self.checkpoint_frame_fn(draws)
        encoded = _pickle.dumps(frame, protocol=_pickle.HIGHEST_PROTOCOL)
        if len(encoded) > self.checkpoint_frame_max_bytes:
            raise ValueError(
                "checkpoint visualization frame is "
                f"{len(encoded)} bytes; limit is {self.checkpoint_frame_max_bytes}"
            )
        state = "partial" if partial else "final"
        path = os.path.join(
            ckpt_dir,
            f"ns_checkpoint_frame_{os.getpid()}_"
            f"{int(progress['iteration']):06d}_{state}.pkl",
        )
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(encoded)
        os.replace(tmp, path)

        pattern = os.path.join(
            ckpt_dir, f"ns_checkpoint_frame_{os.getpid()}_*.pkl"
        )
        for old_path in sorted(_glob.glob(pattern))[:-self.checkpoint_frame_limit]:
            os.remove(old_path)
        return path

    def _dump_snapshot(self, ckpt_dir, live, dead_list, ns_utils, logZ,
                       *, tag, partial, progress):
        """Atomically pickle a finalised snapshot of the run so far.

        Schema 2 adds progress metadata to the legacy recovery keys.  A
        ``partial=True`` snapshot marks a run that had not converged.
        Best-effort: never let a checkpoint break the run.  Atomic via
        write-to-temp + os.replace so a kill mid-write cannot corrupt an
        existing checkpoint.
        """
        try:
            import pickle as _pickle
            import numpy as _np
            pos, logl, logl_birth = self._finalise_dead(live, dead_list,
                                                        ns_utils)
            os.makedirs(ckpt_dir, exist_ok=True)
            fname = (f"ns_raw_dead_{os.getpid()}.pkl" if tag == "rescue"
                     else f"ns_checkpoint_{os.getpid()}.pkl")
            path = os.path.join(ckpt_dir, fname)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                snapshot = {
                    "schema_version": 2,
                    "positions": {k: _np.asarray(v) for k, v in pos.items()},
                    "loglikelihood": _np.asarray(logl),
                    "loglikelihood_birth": _np.asarray(logl_birth),
                    "logZ": float(logZ),
                    "n_dead": int(_np.asarray(logl).shape[0]),
                    "partial": bool(partial),
                    "progress": dict(progress),
                }
                _pickle.dump(snapshot, fh)
            os.replace(tmp, path)
            try:
                self._dump_checkpoint_frame(
                    ckpt_dir, snapshot, progress, partial=partial
                )
            except Exception as exc:                              # noqa: BLE001
                print(f"  [{tag}] WARNING: visualization frame failed: {exc}")
            return path
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [{tag}] WARNING: snapshot failed: {exc}")
            return None

    @staticmethod
    def load_checkpoint(path):
        """Load a checkpoint / rescue pickle written by this adapter.

        Returns the dict ``{positions, loglikelihood, loglikelihood_birth,
        logZ, n_dead, partial}``.  A ``partial=True`` snapshot is a usable
        (under-converged) posterior from a run killed before convergence ---
        feed ``positions`` + ``loglikelihood`` + ``loglikelihood_birth`` to
        ``anesthetic.NestedSamples`` exactly as the end-of-run path does.
        """
        import pickle as _pickle
        with open(path, "rb") as fh:
            return _pickle.load(fh)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _build_nested_sampler(self, loglike_fn, logprior_fn,
                              num_inner_steps, num_delete):
        """Construct the BlackJAX NSS transition kernel.

        ``slice_kernel="carry"`` assembles the kernel as
        ``blackjax.ns.nss.as_top_level_api`` does, with
        :func:`stepping_out_carry` as the interval procedure.
        ``"lanes"`` replaces the inner update with :func:`lane_update`.
        """
        import blackjax
        if self.slice_kernel == "stock":
            return blackjax.nss(
                logprior_fn=logprior_fn, loglikelihood_fn=loglike_fn,
                num_delete=num_delete, num_inner_steps=num_inner_steps,
            )
        from functools import partial
        from blackjax.mcmc.slice import build_kernel as build_slice_kernel
        from blackjax.ns import adaptive, base, from_mcmc, nss
        init_state_fn = partial(
            base.init_state_strategy,
            logprior_fn=logprior_fn, loglikelihood_fn=loglike_fn)
        if self.slice_kernel == "lanes":
            kernel = adaptive.build_kernel(
                partial(base.delete_fn, num_delete=num_delete),
                lane_update(logprior_fn, loglike_fn, num_inner_steps, num_delete),
                update_inner_kernel_params_fn=nss.live_covariance)
        else:
            constrained_step = nss.slice_constrained_step(
                init_state_fn,
                build_slice_kernel(interval=stepping_out_carry,
                                   max_expansions=10, max_shrinkage=100),
                nss.covariance_proposal)
            kernel = from_mcmc.build_kernel(
                constrained_step, num_inner_steps, nss.live_covariance,
                num_delete)

        def init_fn(position, rng_key=None):
            return adaptive.init(
                position, init_state_fn=jax.vmap(init_state_fn),
                update_inner_kernel_params_fn=nss.live_covariance,
                rng_key=rng_key)

        return blackjax.SamplingAlgorithm(init_fn, kernel)

    @staticmethod
    def _logical_likelihood_calls(info):
        """Per-particle evaluations for the pinned stepping-out slice kernel.

        Each expansion loop evaluates its terminating condition once more
        than its body runs. The two endpoints therefore add two calls per
        slice. This counts logical evaluations, not redundant GPU lanes in
        a vectorized while loop.
        """
        return jnp.sum(info.update_info.num_expansions
                       + info.update_info.num_shrink + 2)

    def run(
        self,
        loglike_fn  : Callable[[dict[str, Array]], Array],
        logprior_fn : Callable[[dict[str, Array]], Array],
        theta_init  : dict[str, Array],
        rng_key     : Array,
    ) -> SamplingResult:
        """
        Run BlackJAX NSS and return a ``SamplingResult``.

        Parameters
        ----------
        loglike_fn : callable
            JIT-compiled log-likelihood (no prior).
        logprior_fn : callable
            JIT-compiled log-prior (must be proper).
        theta_init : dict[str, Array]
            Reference parameter dict (shapes / dtypes).
        rng_key : Array

        Returns
        -------
        SamplingResult
        """
        try:
            import blackjax
            import blackjax.ns.utils as ns_utils
        except ImportError as exc:
            raise ImportError(
                "BlackJAX with nested sampling (blackjax.ns) is required.\n"
                "Install: pip install 'git+https://github.com/blackjax-devs/blackjax@f73e12956'"
            ) from exc

        import tqdm

        n_dims          = self._n_dims(theta_init)
        num_inner_steps = (self._num_inner_steps
                           if self._num_inner_steps is not None
                           else n_dims * 5)
        num_delete      = (self._num_delete
                           if self._num_delete is not None
                           else max(1, self.num_live // 5))

        if self.verbose:
            print(
                f"BlackJAX NSS  |  n_dims={n_dims}  "
                f"num_live={self.num_live}  "
                f"num_inner_steps={num_inner_steps}  "
                f"num_delete={num_delete}"
            )

        # ── Initialise live points ────────────────────────────────────────
        rng_key, prior_key = jax.random.split(rng_key)
        particles = self._sample_prior(theta_init, prior_key)

        # ── Build NSS kernel ──────────────────────────────────────────────
        # loglike_fn / logprior_fn operate on a SINGLE particle (un-batched).
        # The NSS step_fn vmaps internally over the live-point ensemble.
        nested_sampler = self._build_nested_sampler(
            loglike_fn, logprior_fn, num_inner_steps, num_delete)
        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)

        if self.verbose:
            print("  [timing] Calling init_fn (JIT compile + eval) and "
                  "compiling the step kernel ...",
                  flush=True)
            _t0 = time.perf_counter()

        # Compile the step kernel in a thread while init compiles and runs;
        # the first step_fn call then takes the executable from JAX's cache.
        live_shape = jax.eval_shape(init_fn, particles)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            step_compiled = pool.submit(
                lambda: step_fn.lower(rng_key, live_shape).compile())
            live = init_fn(particles)
            step_compiled.result()

        # ── BlackJAX version compatibility ───────────────────────────────
        # Three known layouts for logZ / logZ_live:
        #   v1  NSState (direct):           state.logZ, state.logZ_live
        #   v2  AdaptiveNSState (wrapper):  state.sampler_state.logZ, ...
        #   v3  AdaptiveNSState (integrator): state.integrator.logZ, ...
        def _build_logZ_accessors(state):
            """Return (get_logZ, get_logZ_live) callables for *state*."""
            # v1 – direct NSState
            if hasattr(state, "logZ") and hasattr(state, "logZ_live"):
                return (lambda s: float(s.logZ),
                        lambda s: float(s.logZ_live))
            # v3 – newest: integrator sub-object
            if hasattr(state, "integrator"):
                ig = state.integrator
                if hasattr(ig, "logZ") and hasattr(ig, "logZ_live"):
                    return (lambda s: float(s.integrator.logZ),
                            lambda s: float(s.integrator.logZ_live))
            # v2 – older wrapper: sampler_state sub-object
            if hasattr(state, "sampler_state"):
                inner = state.sampler_state
                if hasattr(inner, "logZ") and hasattr(inner, "logZ_live"):
                    return (lambda s: float(s.sampler_state.logZ),
                            lambda s: float(s.sampler_state.logZ_live))
            # Unknown layout – raise with diagnostics
            _fields = [f for f in dir(state) if not f.startswith("_")]
            raise AttributeError(
                f"Cannot locate logZ/logZ_live on {type(state).__name__}.\n"
                f"  Top-level fields : {_fields}\n"
                + "Please check your BlackJAX version."
            )

        _get_logZ, _get_logZ_live = _build_logZ_accessors(live)

        if self.verbose:
            _t1 = time.perf_counter()
            print(f"  [timing] init_fn and step compile done    ({_t1 - _t0:.1f} s)  "
                  f"logZ={_get_logZ(live):.4f}  "
                  f"logZ_live={_get_logZ_live(live):.4f}", flush=True)
            print(f"  (state type: {type(live).__name__})")

        # ── NS run loop ───────────────────────────────────────────────────
        dead_list    = []
        n_like_calls = self.num_live
        t_start      = time.perf_counter()

        # Periodic-checkpoint bookkeeping (see __init__).
        _ckpt_dir   = self._resolve_ckpt_dir()
        _ckpt_on    = bool(_ckpt_dir) and self.checkpoint_interval_s > 0
        _last_ckpt  = t_start
        if _ckpt_on and self.verbose:
            print(f"  [checkpoint] every {self.checkpoint_interval_s:.0f} s "
                  f"-> {_ckpt_dir}", flush=True)

        def _progress_record(logZ, logZ_live, elapsed_s):
            return {
                "iteration": _iter,
                "n_likelihood_calls": n_like_calls,
                "n_discarded": num_delete * _iter,
                "n_live": self.num_live,
                "num_delete": num_delete,
                "num_inner_steps": num_inner_steps,
                "logZ": logZ,
                "logZ_live": logZ_live,
                "delta_logZ": logZ_live - logZ,
                "elapsed_s": elapsed_s,
            }

        _progress_file = (open(self.progress_path, "a", encoding="utf-8")
                          if self.progress_path else None)

        desc = "NS  (starting)"
        try:
            desc = f"NS  logZ={_get_logZ(live):.1f}"
        except (AttributeError, NameError):
            pass

        with tqdm.tqdm(
            desc=desc,
            unit=" dead",
            disable=not self.verbose,
        ) as pbar:
            _iter = 0
            # Reuse evidence reads for the stop condition and progress output.
            # Slice counters are read separately for logical call accounting.
            _logZ = _get_logZ(live)
            _logZ_live = _get_logZ_live(live)
            while float(_logZ_live - _logZ) >= self.logZ_tol:
                rng_key, subkey = jax.random.split(rng_key)
                _t_iter = time.perf_counter()

                incoming = live
                live, dead_info = step_fn(subkey, incoming)
                _iter += 1
                _logZ = _get_logZ(live)
                _logZ_live = _get_logZ_live(live)
                _dt_iter = time.perf_counter() - _t_iter
                if self.iteration_callback is not None:
                    self.iteration_callback(
                        _iter, subkey, incoming, live, dead_info,
                        step_fn, _dt_iter)
                if self.verbose:
                    print(
                        f"  [iter {_iter:>4d}]  {_dt_iter:6.1f} s  "
                        f"logZ={_logZ:+.3f}  ΔlogZ={_logZ_live - _logZ:.3f}  "
                        f"dead={num_delete * _iter}",
                        flush=True,
                    )
                dead_list.append(dead_info)
                n_like_calls += int(self._logical_likelihood_calls(dead_info))
                if _progress_file is not None:
                    _elapsed = time.perf_counter() - t_start
                    _record = _progress_record(_logZ, _logZ_live, _elapsed)
                    _record["iteration_s"] = _dt_iter
                    _record["dead_per_s"] = _record["n_discarded"] / _elapsed
                    _record["likelihood_calls_per_s"] = n_like_calls / _elapsed
                    _progress_file.write(json.dumps(_record) + "\n")
                    _progress_file.flush()
                pbar.update(num_delete)
                try:
                    pbar.set_description(
                        f"NS  logZ={_logZ:.2f}  "
                        f"ΔlogZ={_logZ_live - _logZ:.2f}"
                    )
                except AttributeError:
                    pass

                # Periodic checkpoint: finalise + dump a partial snapshot so a
                # wall-time kill / node crash mid-run is recoverable.
                if _ckpt_on and (time.perf_counter() - _last_ckpt
                                 >= self.checkpoint_interval_s):
                    _progress = _progress_record(
                        _logZ, _logZ_live, time.perf_counter() - t_start)
                    _p = self._dump_snapshot(
                        _ckpt_dir, live, dead_list, ns_utils,
                        _logZ, tag="checkpoint", partial=True,
                        progress=_progress)
                    _last_ckpt = time.perf_counter()
                    if _p and self.verbose:
                        print(f"  [checkpoint] iter {_iter}: {_p}", flush=True)

        wall_time = time.perf_counter() - t_start
        if _progress_file is not None:
            _progress_file.close()
        if self.verbose:
            print(
                f"  Converged  logZ = {_get_logZ(live):.3f}  "
                f"({wall_time:.1f} s,  {n_like_calls:,} likelihood calls)"
            )

        # ── Merge live points into dead set ───────────────────────────────
        # Version-aware finalise + dead-point unpack (see _finalise_dead).
        # The ``update_info`` kwarg only exists in newer BlackJAX; 0.1.0b0
        # ships ``finalise(live, dead)`` with no such param, so passing it
        # unconditionally raises ``TypeError`` only AFTER convergence,
        # losing a multi-hour run.  The guard lives in _finalise_dead so it
        # (and the periodic-checkpoint path) cannot drift.
        _dead_positions, _dead_logl, _dead_logl_birth = self._finalise_dead(
            live, dead_list, ns_utils)

        # ── Rescue pickle ─────────────────────────────────────────────────
        # Dump the finalised dead points so any failure further down the save
        # path (anesthetic, evidence, the caller's I/O) is recoverable rather
        # than discarding a multi-hour run.  Same format as the periodic
        # checkpoints; loadable via load_checkpoint().  partial=False marks a
        # fully-converged snapshot.
        _rescue_dir = self._resolve_ckpt_dir()
        if _rescue_dir:
            _progress = _progress_record(
                _get_logZ(live), _get_logZ_live(live), wall_time)
            self._dump_snapshot(_rescue_dir, live, dead_list, ns_utils,
                                _get_logZ(live), tag="rescue", partial=False,
                                progress=_progress)

        # ── Evidence & importance weights (anesthetic preferred) ─────────
        log_Z        = float(_get_logZ(live))
        log_Z_err    = float("nan")
        log_weights  = None
        try:
            from anesthetic import NestedSamples
            import numpy as np

            _raw   = _dead_positions
            _names = [n for n in _raw if n in theta_init]
            _cols  = [
                np.asarray(_raw[n]).reshape(
                    len(np.asarray(_dead_logl)), -1
                )
                for n in _names
            ]
            _data  = np.hstack(_cols)
            _ns    = NestedSamples(
                _data,
                logL       = np.asarray(_dead_logl),
                logL_birth = np.asarray(_dead_logl_birth),
                logzero    = float("nan"),
            )
            log_Z     = float(_ns.logZ())
            log_Z_err = float(_ns.logZ(12).std())
            # Extract proper NS importance weights from anesthetic.
            # These encode the prior-volume compression at each dead point.
            log_weights = jnp.asarray(np.asarray(_ns.logw()))
        except Exception:
            pass  # fall back to live.logZ and manual weights below

        # Fallback: compute log-weights from prior volume shrinkage if
        # anesthetic is unavailable or failed.
        if log_weights is None:
            n_dead  = len(jnp.asarray(_dead_logl))
            n_live  = self.num_live
            # Standard NS trapezoid rule: log(X_{i-1} - X_{i+1}) / 2
            # where X_i = exp(-i / n_live) is the prior volume fraction.
            log_vols = -jnp.arange(n_dead, dtype=float) / n_live
            log_dvol = jnp.log(
                jnp.exp(jnp.roll(log_vols, 1) - log_vols)
                - jnp.exp(jnp.roll(log_vols, -1) - log_vols)
            ) + log_vols
            # Fix boundary: first and last points
            log_dvol = log_dvol.at[0].set(
                jnp.log1p(-jnp.exp(-1.0 / n_live))
            )
            log_dvol = log_dvol.at[-1].set(
                log_vols[-1] - jnp.log(n_live)
            )
            log_weights = log_dvol

        # ── Pack samples ──────────────────────────────────────────────────
        # Squeeze trailing size-1 axes so scalar params have shape (n_dead,)
        # rather than (n_dead, 1), matching typical user expectation.
        samples = {}
        for name in theta_init:
            arr = jnp.asarray(_dead_positions[name])   # (n_dead, *shape)
            # Squeeze only if the parameter was a scalar (shape (1,))
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = jnp.squeeze(arr, axis=-1)
            samples[name] = arr

        return SamplingResult(
            samples               = samples,
            log_evidence          = log_Z,
            log_evidence_err      = log_Z_err,
            log_weights           = log_weights,
            log_likelihoods       = jnp.asarray(_dead_logl),
            log_likelihoods_birth = jnp.asarray(_dead_logl_birth),
            param_names           = list(theta_init.keys()),
            n_likelihood_calls    = n_like_calls,
            wall_time_s           = wall_time,
            sampler_name          = "blackjax.nss",
            likelihood_count_kind = "logical_including_initialization",
            raw                   = {
                "positions": _dead_positions,
                "loglikelihood": _dead_logl,
                "loglikelihood_birth": _dead_logl_birth,
            },
        )
