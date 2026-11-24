"""Offline checkpoint-selection metrics for the policy.

WHY THIS EXISTS
---------------
`select_metric` used to name an environment-rollout metric, so the checkpoint handed to the next
DAgger round was chosen by running the policy in the simulator -- the same simulator that produces
the reported number. Every metric in this module is computed from stored data only: no environment,
no rollout, no planner. So selection can no longer see the thing being reported.

WHAT IT MEASURES, AND WHY NOT JUST THE VALIDATION LOSS
-----------------------------------------------------
The literature is consistent that an averaged validation loss ranks checkpoints badly (see
CODE_AUDIT-adaBot.md Part 14.3). Two specific criticisms shaped what is computed here.

1. *Not all action error matters equally.* The policy executes element 0 of the last predicted
   chunk and discards the rest (train_policy.py's rollout loop). An average over all nine action
   slots in a window therefore mixes the one number that reaches the environment with eight that
   never do. `exec_mse` isolates the executed action; `chunk_mse` keeps the full-chunk view for
   comparison.

2. *A mean hides the failures that actually end an episode.* A policy that is slightly wrong
   everywhere and a policy that is catastrophically wrong on 5% of states can have the same mean
   error and very different success rates. So every error metric here also reports its 90th
   percentile and the fraction of windows above a threshold.

THE FROZEN HOLDOUT
------------------
These numbers are only comparable across rounds if the data they are computed on does not change.
The training run's own validation split is drawn from a pool that grows every round, so its
membership shifts and its numbers are not a yardstick. `select_holdout_path` must therefore point
at a FIXED directory of planned shards that is never trained on and never added to.
"""

import numpy as np
import torch


def _tail_stats(errs, threshold):
    """mean, 90th percentile, and the fraction above `threshold`, over per-window errors."""
    a = np.asarray(errs, dtype=np.float64)
    if a.size == 0:
        return {}
    out = {"mean": float(a.mean()), "p90": float(np.percentile(a, 90))}
    if threshold is not None and threshold > 0:
        out["frac_bad"] = float((a > threshold).mean())
    return out


@torch.no_grad()
def offline_metrics(run_model, loader, model, device, err_threshold=None, max_batches=None,
                    sample_seed=12345):
    """Offline selection metrics for one holdout loader.

    run_model: the training script's own forward closure, so observation embedding, goal handling
               and precompute behave exactly as in training.
    model:     the UNWRAPPED policy, for `_unpack_actions` (the dataset stores actions packed as
               (N, window + action_window - 1, A) and the model's loss works on (N, T, W, A)).

    sample_seed matters more than it looks. `bet.py:_sample_action` draws the codebook index with
    `torch.multinomial`, not argmax, so the policy's prediction is STOCHASTIC: the same weights on
    the same window give a different action each call. A selection metric has to be a deterministic
    function of the weights, or two checkpoints get compared partly on which one was luckier. So
    the RNG is reseeded to a fixed value before every pass, which makes the sampling identical
    across epochs, rounds and arms while leaving the model untouched. `classification_loss` is the
    one metric here that never touches the sampler, so it is the sampling-free alternative if you
    want selection that does not depend on this at all.

    Returns a flat dict of floats. Keys are stable, so `select_metric` can name any of them.
    """
    model.eval()
    # fixed sampler state: see the sample_seed note above
    _rng_state = torch.get_rng_state()
    _cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(sample_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sample_seed))

    sums, n_batches = {}, 0
    exec_errs, chunk_errs = [], []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        pred, loss, loss_dict = run_model(batch)
        if loss is None or pred is None:
            continue
        n_batches += 1

        # batch-level quantities the model already computes. Accumulated as a running mean, not a
        # running sum: the existing eval/epoch_wise_* series are sums and therefore grow with the
        # number of batches, which makes them incomparable between rounds (bug B6).
        for k in ("total_loss", "offset_loss", "classification_loss",
                  "equal_total_code_rate", "equal_single_code_rate", "equal_single_code_rate2"):
            if k in loss_dict:
                v = loss_dict[k]
                sums[k] = sums.get(k, 0.0) + float(v.item() if torch.is_tensor(v) else v)

        # per-window errors, so the distribution (not just its mean) is available.
        act_true = model._unpack_actions(batch[1].to(device))   # (N, T, W, A)
        # [:, -1, 0] is the action the rollout actually executes; bet.py says so at _calc_loss.
        e_exec = ((act_true[:, -1, 0] - pred[:, -1, 0]) ** 2).mean(dim=-1)
        e_chunk = ((act_true[:, -1] - pred[:, -1]) ** 2).mean(dim=(-1, -2))
        exec_errs.append(e_exec.detach().float().cpu().numpy())
        chunk_errs.append(e_chunk.detach().float().cpu().numpy())

    # restore the training RNG, so reseeding the sampler above cannot shift the data order,
    # dropout masks or anything else the training loop draws afterwards.
    torch.set_rng_state(_rng_state)
    if _cuda_state is not None:
        torch.cuda.set_rng_state_all(_cuda_state)

    if n_batches == 0:
        return {}

    out = {k: v / n_batches for k, v in sums.items()}
    for name, chunks in (("exec_mse", exec_errs), ("chunk_mse", chunk_errs)):
        for stat, val in _tail_stats(np.concatenate(chunks), err_threshold).items():
            out[f"{name}_{stat}" if stat != "mean" else name] = val
    out["n_windows"] = float(sum(len(c) for c in exec_errs))
    # Codebook agreement is a rate, so higher is better. Negate it too, so that every key in this
    # dict is a "lower is better" quantity and select_mode: min is correct for all of them.
    if "equal_total_code_rate" in out:
        out["code_err"] = 1.0 - out["equal_total_code_rate"]
    return out


def offline_metrics_multi(run_model, loaders, model, device,
                          err_threshold=None, max_batches=None, prefix="offline",
                          sample_seed=12345):
    """Run `offline_metrics` over several named holdout loaders.

    Keys come back as "<prefix>/<split>/<metric>", e.g. "offline/success/exec_mse", so
    select_metric can name one split's metric. Splits are the point: RACE argues a score should be
    weighted toward the trajectories that matter, and the cheapest version of that here is to
    report the planner's successes and failures separately rather than averaging them together.
    """
    out = {}
    for split, loader in loaders.items():
        if loader is None:
            continue
        m = offline_metrics(run_model, loader, model, device,
                            err_threshold=err_threshold, max_batches=max_batches,
                            sample_seed=sample_seed)
        for k, v in m.items():
            out[f"{prefix}/{split}/{k}"] = v
    return out
