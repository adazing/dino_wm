"""Systematic audit of the dino_wm DAgger config plumbing.

Looks for the class of bug that made wm_original_weight a ten-round no-op:
  1. ORPHAN ++keys  - dagger_loop pushes `++key=value` into a subprocess, and Hydra's `++`
                      FORCE-ADDS the key. So a key nothing reads is silently a no-op. A typo
                      here cannot fail loudly.
  2. SILENT IGNORES - log lines saying a setting was ignored / overridden. Each is a trap.
  3. DEAD config keys - declared in a dagger config but read by nothing.
Run on the NAS box:  python3 /tmp/audit.py
"""
import glob
import re

SCRIPTS = ("train.py", "train_policy.py", "generate_planned_trajectories.py",
           "policy_rollout.py", "dagger_loop.py")
src = {f: open(f).read() for f in SCRIPTS}
confs = {f: open(f).read() for f in glob.glob("conf/**/*.yaml", recursive=True)}
dl = src["dagger_loop.py"]


def is_read(leaf, files):
    """Is `leaf` read anywhere in `files`?

    Must cope with indirection, which is how most keys are actually read here:
        cfg.get("k") / cfg.k / cfg["k"]        direct
        _cap_of("max_demo_windows")            a bare literal handed to a helper
        (("epochs", "policy_epochs"), ...)     a literal inside a lookup table
        cfg.get(f"max_{src}_windows")          an f-string, so the literal never appears
    The f-string case is matched loosely by prefix + suffix. Loose beats reporting a false
    orphan: an audit that cries wolf gets ignored, which is how the real one slips through.
    """
    for f in files:
        s = src[f]
        if re.search(r"""get\(["']%s["']""" % re.escape(leaf), s):
            return f
        if re.search(r"cfg\.%s\b" % re.escape(leaf), s):
            return f
        if re.search(r"""["']%s["']""" % re.escape(leaf), s):      # any bare literal
            return f
        parts = leaf.split("_")
        if len(parts) >= 3:                                        # f"{pre}_{...}_{suf}"
            pat = r"""f["'][^"']*%s_\{[^}]+\}_%s""" % (re.escape(parts[0]), re.escape(parts[-1]))
            if re.search(pat, s):
                return f
    return None


print("=" * 78)
print("1. ++keys dagger_loop pushes into subprocesses")
print("=" * 78)
def strip_docstrings(text):
    """Drop triple-quoted blocks, so prose showing an example override is not read as one."""
    return re.sub(r'("""|\'\'\')(?:(?!\1).)*\1', "", text, flags=re.S)


# Scan the code only. Emissions appear both as bare list elements and inside `cmd += [f"++..."]`,
# so match anywhere -- but not inside a docstring that happens to document the syntax.
pushed = sorted(set(re.findall(r"\+\+([A-Za-z_][\w.]*)=", strip_docstrings(dl))))
targets = [f for f in SCRIPTS if f != "dagger_loop.py"]
orphans = []
for k in pushed:
    leaf = k.split(".")[-1]
    where = is_read(leaf, targets)
    in_conf = [f for f, s in confs.items() if re.search(r"^\s*%s:" % re.escape(leaf), s, re.M)]
    if not where and not in_conf:
        orphans.append(k)
print(f"   {len(pushed)} distinct keys emitted")
if orphans:
    print("   ORPHANS - nothing reads them and no config declares them:")
    for k in orphans:
        print("      !!", k)
else:
    print("   OK: every key is read by a script or declared in a config")

print()
print("=" * 78)
print("2. nullable ++overrides not wrapped in _hy()")
print("=" * 78)
# Hydra parses "++k=None" as the STRING 'None', never as null. An f-string that interpolates a
# possibly-None value straight into an override therefore ships 'None' downstream, where it fails
# int() or looks "set" to a presence check. Only "++k=null" yields a real None.
unwrapped = []
for m in re.finditer(r'f"\+\+([\w.]+)=\{([^}]*)\}"', dl):
    key, expr = m.group(1), m.group(2)
    nullable = ("None)" in expr) or expr.strip() == "None"
    if nullable and "_hy(" not in expr:
        unwrapped.append(f"++{key}={{{expr}}}")
if unwrapped:
    print("   NOT WRAPPED - these will ship the string 'None':")
    for u in unwrapped:
        print("      !!", u)
else:
    print("   OK: every nullable override goes through _hy()")

print()
print("=" * 78)
print("3. silent-ignore / precedence log lines (each one is a trap)")
print("=" * 78)
found = 0
for f in SCRIPTS:
    for line in src[f].splitlines():
        t = line.strip()
        if t.startswith("#"):
            continue
        if re.search(r"ignored|only applies|has no effect|takes precedence", t):
            print(f"   {f}: {t[:108]}")
            found += 1
if not found:
    print("   OK: none left")

print()
print("=" * 78)
print("4. dagger config keys that nothing reads")
print("=" * 78)
KNOWN_INFRA = {"defaults", "hydra", "out_dir", "env", "dataset", "encoder", "policy",
               "wandb_project", "wandb_group", "experiment"}
for cf in ("conf/dagger_wall.yaml", "conf/dagger_pusht.yaml"):
    dead = []
    for m in re.finditer(r"^([a-z_][a-z0-9_]*):", confs[cf], re.M):
        key = m.group(1)
        if key in KNOWN_INFRA:
            continue
        # read by dagger_loop itself, or forwarded, or read downstream
        if re.search(r"""get\(["']%s["']""" % key, dl) or re.search(r"cfg\.%s\b" % key, dl):
            continue
        if re.search(r"""f?["']\+\+%s=""" % key, dl):
            continue
        # read indirectly by dagger_loop, e.g. the (flag, cfg_key) passthrough table
        if re.search(r"""["']%s["']""" % key, dl):
            continue
        if is_read(key, targets):
            continue
        dead.append(key)
    print(f"   {cf}: " + (", ".join(dead) if dead else "OK: every key is read"))

print()
print("=" * 78)
print("5. callers of rollout_policy (signature changed this session)")
print("=" * 78)
for f in SCRIPTS + ("eval_yardsticks.py",):
    try:
        s = open(f).read()
    except OSError:
        continue
    for m in re.finditer(r"rollout_policy\(", s):
        ln = s[:m.start()].count("\n") + 1
        if "def rollout_policy" in s[max(0, m.start() - 20):m.start()]:
            continue
        print(f"   {f}:{ln}  (positional args unchanged; record_* are keyword-only additions)")
