"""Test-time environment shifts: layout overrides and observation corruptions.

Everything here is OFF by default. Nothing in this module changes behaviour unless a shift
is explicitly configured, so existing runs are bit-identical.

There are two separate mechanisms, because the two kinds of shift enter the pipeline at
different places.

APPEARANCE SHIFTS -> constructor kwargs, via `env.kwargs`
---------------------------------------------------------
Colours and image corruptions are properties of the environment object, so they are passed
to the constructor and `gym.make(name, **cfg.env.kwargs)` carries them to every consumer:
the CEM planner during generation, the policy evaluation that produces the metric,
policy_rollout, and the recorded videos. No per-call-site plumbing, and the videos show
exactly what the model was shown.

  wall     ++env.kwargs.wall_color=[255,0,0]        black bar   -> red
           ++env.kwargs.agent_color=[0,0,255]       red dot     -> blue
           ++env.kwargs.bg_color=[200,200,200]      white field -> grey
  pusht    ++env.kwargs.shape=I                     block geometry, see pusht_env.add_shape
           ++env.kwargs.color=red                   moving block, AdaJEPA's "redBlock"
           ++env.kwargs.agent_color=red             pusher,       AdaJEPA's "redAgent"
           ++env.kwargs.goal_color=red              anchor T,     AdaJEPA's "redAnchor"
  both     ++env.kwargs.obs_corruption=blur         blur | snp1 | snp5 | dark
           ++env.kwargs.obs_corruption_level=2.0    overrides the default level

LAYOUT SHIFTS -> `eval_env_override`
------------------------------------
Wall position, door position and pusht shape are re-applied from the DATASET at evaluation
and generation time: `wall_dset.py:115` returns the trajectory's own door/wall location and
`pusht_dset.py:171` returns its shape, and those go into `env.update_env(...)`. A
constructor kwarg is therefore silently overwritten, which is why these need an override
that is merged ON TOP of whatever the data supplies:

  wall     ++eval_env_override.fix_wall_location=40 ++eval_env_override.fix_door_location=48
  pusht    ++eval_env_override.shape=I

The keys are whatever that environment's own `update_env` understands, so this module stays
environment-agnostic.

WHY CORRUPTION IS NOT A LAYOUT OVERRIDE
---------------------------------------
A corruption does not change the environment, only what the model is shown: the simulator
state, the dynamics and the success test are untouched. That makes it the cleanest possible
appearance shift, and it is how AdaJEPA's blur/snp/dark experiments are defined. The
functions below are ported from adajepa/planning/image_corruption.py.
"""

import numpy as np

try:
    import torch
except Exception:                                      # pragma: no cover - torch is a hard dep
    torch = None

try:
    import cv2
except Exception:
    cv2 = None


# --------------------------------------------------------------------------------------
# corruptions. Each takes (uint8 HWC frame, level, rng) and returns a uint8 HWC frame.
# --------------------------------------------------------------------------------------

def _gaussian_blur_numpy(frame, sigma):
    """Separable Gaussian blur with edge padding, so this module works without OpenCV.

    cv2.GaussianBlur is used when available; this is the fallback and matches it closely
    (same kernel radius rule, same normalisation, same edge handling).
    """
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    img = frame.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (radius, radius)
        p = np.pad(img, pad, mode="edge")
        acc = np.zeros_like(img)
        for i, w in enumerate(k):
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(i, i + img.shape[axis])
            acc += w * p[tuple(sl)]
        img = acc
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_blur(frame, sigma, rng=None):
    """Gaussian blur. AdaJEPA's default sigma is 2.0."""
    if cv2 is not None:
        k = max(3, int(2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)
    return _gaussian_blur_numpy(frame, sigma)


def apply_salt_pepper(frame, density, rng):
    """Salt-and-pepper noise. `density` is the fraction of pixels set white, and the same
    fraction again set black, so roughly 2*density of the image is replaced."""
    h, w = frame.shape[:2]
    out = frame.copy()
    n = int(density * h * w)
    if n <= 0:
        return out
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 255
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 0
    return out


def apply_dark(frame, factor, rng=None):
    """Darken by multiplying every channel by `factor` < 1. AdaJEPA uses 0.5."""
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# name -> (function, default level). Levels are AdaJEPA's.
CORRUPTIONS = {
    "blur": (apply_blur, 2.0),
    "snp1": (apply_salt_pepper, 0.01),
    "snp5": (apply_salt_pepper, 0.05),
    "dark": (apply_dark, 0.5),
}


def corrupt_visual(arr, name, level=None, seed=0):
    """Corrupt a visual observation, preserving its container, dtype and value scale.

    Accepts a numpy array or a torch tensor with any number of leading batch/time dims and
    a trailing (H, W, C). uint8, float in [0, 1] and float in [0, 255] all round-trip, so
    this can sit anywhere in the pipeline without caring which env produced the frame.

    `seed` makes salt-and-pepper deterministic; pass something that varies per episode but
    is stable across reruns (the env seed) so evaluation is reproducible.
    """
    if not name or name == "none":
        return arr
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown obs_corruption {name!r}; choose from "
                         f"{sorted(CORRUPTIONS)} or 'none'.")
    fn, default_level = CORRUPTIONS[name]
    lvl = float(default_level if level is None else level)

    is_torch = torch is not None and isinstance(arr, torch.Tensor)
    a = arr.detach().cpu().numpy() if is_torch else np.asarray(arr)
    if a.ndim < 3:
        raise ValueError(f"corrupt_visual expects a trailing (H, W, C), got shape {a.shape}")

    orig_dtype = a.dtype
    unit_scale = False                       # was the input float in [0, 1]?
    if orig_dtype == np.uint8:
        u8 = a
    else:
        unit_scale = bool(a.size) and float(np.nanmax(a)) <= 1.5
        u8 = np.clip(a * 255.0 if unit_scale else a, 0, 255).astype(np.uint8)

    flat = u8.reshape(-1, *u8.shape[-3:])
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    done = np.stack([fn(flat[i], lvl, rng) for i in range(flat.shape[0])])
    done = done.reshape(u8.shape)

    if orig_dtype == np.uint8:
        out = done
    elif unit_scale:
        out = (done.astype(np.float64) / 255.0).astype(orig_dtype)
    else:
        out = done.astype(orig_dtype)
    return torch.as_tensor(out) if is_torch else out


def corrupt_obs(obs, name, level=None, seed=0):
    """corrupt_visual applied to obs['visual'], leaving proprio and everything else alone.

    Returns a new dict; the input is not mutated.
    """
    if not name or name == "none" or not isinstance(obs, dict) or "visual" not in obs:
        return obs
    out = dict(obs)
    out["visual"] = corrupt_visual(obs["visual"], name, level, seed)
    return out


# --------------------------------------------------------------------------------------
# colours
# --------------------------------------------------------------------------------------

_NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128),
    "gray": (128, 128, 128), "yellow": (255, 255, 0), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "orange": (255, 165, 0),
}


def parse_rgb(value, default):
    """Accept a name ('red'), an '#rrggbb' string, or a 3-sequence; return an (r, g, b) tuple.

    `default` is returned for None, so a config that does not set a colour keeps the
    environment's original appearance.
    """
    if value is None:
        return tuple(int(c) for c in default)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _NAMED_COLORS:
            return _NAMED_COLORS[s]
        if s.startswith("#") and len(s) == 7:
            return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
        raise ValueError(f"unknown colour {value!r}; use a name from "
                         f"{sorted(_NAMED_COLORS)}, '#rrggbb', or [r, g, b].")
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"a colour needs 3 components, got {value!r}")
    return tuple(int(c) for c in seq)


# --------------------------------------------------------------------------------------
# layout overrides
# --------------------------------------------------------------------------------------

# Keys that an env's update_env reads via `.item()`, so a plain python number would raise
# AttributeError. Values for these are wrapped in a tensor. Anything else (pusht's `shape`,
# a string) is passed through untouched.
_WALL_LOCATION_KEYS = ("fix_wall_location", "fix_door_location")


def parse_override(cfg, env_name=None):
    """Read `eval_env_override` off a config and validate it. Returns a dict or None."""
    raw = cfg.get("eval_env_override", None) if hasattr(cfg, "get") else None
    if raw is None:
        return None
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"eval_env_override must be a mapping, got {type(raw).__name__}")
    ovr = {k: v for k, v in raw.items() if v is not None}
    if not ovr:
        return None
    # wall's update_env reads BOTH location keys unconditionally, so supplying one without
    # the other crashes with a bare KeyError inside the env worker. Catch it here, where the
    # message can say what to do.
    if str(env_name) == "wall":
        have = [k for k in _WALL_LOCATION_KEYS if k in ovr]
        if len(have) == 1:
            raise ValueError(
                f"eval_env_override set {have[0]} but not "
                f"{[k for k in _WALL_LOCATION_KEYS if k not in ovr][0]}. wall's update_env "
                f"reads both, so set both -- repeat the trained value for the one you are "
                f"not shifting (wall 32, door 30).")
    return ovr


def apply_override(env_infos, override, n_envs):
    """Merge `override` on top of each per-episode env_info.

    `env_infos` is what the dataset supplied: a list of dicts, a list of Nones, or empty.
    Returns a list of exactly n_envs dicts when an override is active, because
    SerialVectorEnv.update_env indexes env_info[i] for every env and a short list is an
    IndexError. With no override the input is returned unchanged, so the no-shift path is
    untouched.
    """
    if not override:
        return list(env_infos) if env_infos else env_infos
    base = list(env_infos) if env_infos else []
    base = (base + [None] * n_envs)[:n_envs]
    out = []
    for ei in base:
        d = dict(ei) if isinstance(ei, dict) else {}
        for k, v in override.items():
            if k in _WALL_LOCATION_KEYS and torch is not None and not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            d[k] = v
        out.append(d)
    return out


def describe(override, corruption=None, level=None):
    """One-line human summary for the eval banner, so a shifted run is obvious in the log."""
    bits = []
    if override:
        bits.append("env_override=" + ", ".join(
            f"{k}={v.item() if torch is not None and isinstance(v, torch.Tensor) else v}"
            for k, v in sorted(override.items())))
    if corruption and corruption != "none":
        lvl = CORRUPTIONS[corruption][1] if level is None else level
        bits.append(f"obs_corruption={corruption}@{lvl}")
    return "  ".join(bits) if bits else "no shift"
