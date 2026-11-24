"""Weight averaging for the policy, as an alternative to selecting a checkpoint.

WHY THIS EXISTS
---------------
`select_metric` picks the epoch that scored best on some metric. Scoring on
`goal25_valid/max_success_rate` means scoring by simulator rollout, which is the leak recorded as
M2 in CODE_AUDIT-adaBot.md. The obvious replacement is an offline metric, but measured over 39
rounds of four wall arms the held-out action loss correlates with success at Spearman only -0.31,
so no offline metric available here is a trustworthy selector.

Weight averaging sidesteps the problem instead of solving it: there is no selector, so there is
nothing to leak and nothing to validate. It is judged only by the final reported number.

WHY NOT REUSE models/diffusion_policy/diffusion_ema.py
------------------------------------------------------
`EMAModel.step` walks `module.parameters(recurse=False)` and so touches PARAMETERS ONLY, while its
`averaged_model` is a `copy.deepcopy` taken when the EMA was constructed. VQ-BeT keeps its codebook
in BUFFERS -- `register_buffer('embed', ...)`, plus `initted`, `cluster_size`, `embed_avg`,
`codebook_mean` and friends in
models/vq_behavior_transformer/vector_quantize_pytorch/vector_quantize_pytorch.py:289-314. Those
buffers are written during the residual VQ-VAE's fit (`vqvae_fit_steps`). An EMA constructed before
that fit would therefore carry an UNINITIALISED codebook forever, and nothing in `step()` would
ever correct it. That is a silent, severe failure, so this module works on `state_dict()` and is
explicit about what it averages:

  * float parameters that require grad  -> averaged
  * everything else (buffers, integer and bool tensors, frozen parameters) -> copied from the LIVE
    model at save time, never averaged

So the codebook is always the real, fitted codebook.

TWO MODES
---------
  uniform : equal weight to every snapshot taken. This is SWA (Izmailov et al. 2018), which
            averages along a single training trajectory. Appropriate here because
            `warm_start: true` makes each round a fine-tune trajectory from the previous round's
            weights, so the snapshots lie in one basin.
  ema     : exponentially weighted, recent snapshots count more, using the same decay schedule
            shape as the diffusion path's EMAModel (1 - (1 + step/inv_gamma) ** -power). Needs no
            checkpoint files on disk at all, which matters because `save_every=50` over
            `epochs=150` writes only three checkpoints -- too few to average.

NOT GREEDY SOUPING. A greedy soup adds a checkpoint only when it improves a validation score, which
would reintroduce the selector this module exists to avoid.
"""
import torch


class WeightAverager:
    def __init__(self, mode="uniform", ema_power=2.0 / 3.0, ema_inv_gamma=1.0,
                 ema_max_decay=0.9999):
        mode = str(mode).lower()
        if mode not in ("uniform", "ema"):
            raise ValueError("mode must be 'uniform' or 'ema', got %r" % (mode,))
        self.mode = mode
        self.ema_power = float(ema_power)
        self.ema_inv_gamma = float(ema_inv_gamma)
        self.ema_max_decay = float(ema_max_decay)
        self.n = 0                  # snapshots taken
        self._avg = {}              # name -> averaged tensor, float params only
        self._names = None          # which state_dict keys are averaged

    @staticmethod
    def _averageable(model):
        """Float parameters that require grad. Buffers and frozen tensors are excluded."""
        out = set()
        for name, p in model.named_parameters():
            if p.requires_grad and torch.is_floating_point(p):
                out.add(name)
        return out

    def _decay(self):
        """EMA weight on the running average. Matches diffusion_ema.EMAModel.get_decay's shape."""
        step = max(0, self.n - 1)
        if step <= 0:
            return 0.0
        v = 1.0 - (1.0 + step / self.ema_inv_gamma) ** (-self.ema_power)
        return float(min(max(v, 0.0), self.ema_max_decay))

    @torch.no_grad()
    def update(self, model):
        """Fold the model's current weights into the running average."""
        if self._names is None:
            self._names = self._averageable(model)
            if not self._names:
                raise RuntimeError("WeightAverager found no float parameters requiring grad")
        sd = model.state_dict()
        self.n += 1
        if self.n == 1:
            self._avg = {k: sd[k].detach().clone().float() for k in self._names}
            return
        if self.mode == "uniform":
            # running mean: avg += (x - avg) / n
            w = 1.0 / float(self.n)
            for k in self._names:
                self._avg[k].add_(sd[k].detach().float().sub(self._avg[k]), alpha=w)
        else:
            d = self._decay()
            for k in self._names:
                self._avg[k].mul_(d).add_(sd[k].detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def state_dict(self, model):
        """The averaged weights, with every non-averaged entry taken from the LIVE model.

        That is what keeps the VQ codebook and its bookkeeping buffers correct: they are copied
        from the model that actually fitted them, never averaged.
        """
        if self.n == 0:
            raise RuntimeError("WeightAverager.state_dict called before any update()")
        out = {}
        for k, v in model.state_dict().items():
            if k in self._avg:
                out[k] = self._avg[k].to(dtype=v.dtype, device=v.device).clone()
            else:
                out[k] = v.detach().clone()
        return out

    def report(self, model):
        """One line for the training log, plus how far the average sits from the live weights."""
        if self.n == 0:
            return "weight_avg: no snapshots"
        sd = model.state_dict()
        num = den = 0.0
        for k in self._avg:
            a = self._avg[k]
            b = sd[k].detach().float()
            num += float(torch.sum((a - b) ** 2))
            den += float(torch.sum(b ** 2))
        rel = (num ** 0.5) / max(den ** 0.5, 1e-12)
        return ("weight_avg: mode=%s snapshots=%d averaged_tensors=%d "
                "relative_distance_from_live=%.4f"
                % (self.mode, self.n, len(self._avg), rel))
