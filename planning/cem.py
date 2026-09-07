import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        stop_loss_thresh=None,   # opt-in per-traj early-stop, freeze a trajectory once its best
                                 # imagined objective <= this, WM thinks that plan reached goal
        patience=None,   # opt-in per-traj early-stop, freeze a trajectory after this
                                 # many opt-steps with no improvement in its own objective
        min_delta=0.0,   # min per-traj loss improvement to reset its patience counter
        eval_save_plot=True,   # per-opt-step evaluator saves comparison figures (plan.py)
                                 # generation sets False so the real-success check stays cheap
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.stop_loss_thresh = stop_loss_thresh
        self.patience = patience
        self.min_delta = float(min_delta)
        self.eval_save_plot = bool(eval_save_plot)
        # added to the logged "step" so callers (generation) can place each plan() on a single
        # continuous wandb axis (e.g.
        self.log_step_offset = 0

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        # per-trajectory early-stop state, freeze a trajectory once it converges, its imagined
        # objective plateaus for `patience` steps, or drops below `stop_loss_thresh`.
        done = [False] * n_evals   # frozen (converged) flag per trajectory
        best_loss = [float("inf")] * n_evals   # best imagined objective seen (patience)
        stale = [0] * n_evals   # opt-steps since last improvement (patience)
        last_loss = [float("inf")] * n_evals   # most recent per-traj loss (for mean logging)
        for i in range(self.opt_steps):
            # optimize individual instances
            for traj in range(n_evals):
                if done[traj]:
                    continue   # frozen, keep its converged mu/sigma, skip the rollout (the speedup)
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]   # optional, make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                cur_loss = loss[topk_idx[0]].item()
                last_loss[traj] = cur_loss
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                # per-trajectory freeze checks (imagined objective only -> ~free, real success
                if self.stop_loss_thresh is not None and cur_loss <= self.stop_loss_thresh:
                    done[traj] = True   # WM thinks this plan reached the goal
                if self.patience is not None:
                    if cur_loss < best_loss[traj] - self.min_delta:
                        best_loss[traj], stale[traj] = cur_loss, 0
                    else:
                        stale[traj] += 1
                        if stale[traj] >= self.patience:
                            done[traj] = True   # this trajectory's objective plateaued

            mean_loss = float(np.mean(last_loss))
            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": mean_loss,
                 f"{self.logging_prefix}/active": int(n_evals - sum(done)),
                 "step": self.log_step_offset + i + 1}
            )
            if all(done):
                break   # every trajectory converged -> stop optimizing early
            if self.evaluator is not None and self.eval_every is not None and i % self.eval_every == 0:
                # per-trajectory real-success check, roll the current plan out in the ENV and
                # freeze any trajectory the env confirms reached the goal, so it stops costing
                # rollouts.
                with torch.no_grad():
                    logs, successes, _, _ = self.evaluator.eval_actions(
                        mu, save_plot=self.eval_save_plot, save_video=False,
                        filename=f"{self.logging_prefix}_output_{i+1}"
                    )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": self.log_step_offset + i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                for j in range(n_evals):
                    if successes[j]:
                        done[j] = True   # env-confirmed success -> freeze this trajectory
                if all(done):
                    break   # every trajectory reached the goal -> stop planning

        return mu, np.full(n_evals, np.inf)   # all actions are valid
