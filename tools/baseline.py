"""The null model: what does an UNTRAINED policy score on these metrics?

Every round's training logs its environment evaluation at epoch 0, BEFORE that round trains
(train_policy.py runs the eval at `epoch == resume_epoch`). For the FIRST round of an arm the
weights at epoch 0 are the untrained initialisation, so that number is the floor the metric gives
away for free. Any reported success at or near that floor is not evidence the policy learned
anything.

Rounds after the first are warm-started, so their epoch-0 value is the previous round's policy,
not a baseline -- they are printed for context and marked.

usage: python baseline.py $SCRATCH/dagger/<arm> ...
"""
import glob
import json
import os
import sys


def runs(arm):
    from wandb.sdk.internal import datastore as wds
    from wandb.proto import wandb_internal_pb2 as pb
    out = []
    for d in sorted(glob.glob(os.path.join(arm, "wandb", "*run-*"))):
        f = sorted(glob.glob(os.path.join(d, "*.wandb")))
        if not f:
            continue
        rows = {}
        try:
            ds = wds.DataStore()
            ds.open_for_scan(f[0])
        except Exception as e:
            print("   skip %s: %s" % (os.path.basename(d), e))
            continue
        while True:
            try:
                data = ds.scan_data()
            except Exception:
                break
            if data is None:
                break
            rec = pb.Record()
            try:
                rec.ParseFromString(data)
            except Exception:
                continue
            if not rec.HasField("history"):
                continue
            g = {}
            for it in rec.history.item:
                k = it.key or ".".join(it.nested_key)
                if k == "epoch" or "max_success_rate" in k:
                    try:
                        g[k] = float(json.loads(it.value_json))
                    except Exception:
                        pass
            if "epoch" in g and len(g) > 1:
                rows.setdefault(int(g.pop("epoch")), {}).update(g)
        if rows and any("env/" in k for v in rows.values() for k in v):
            out.append((os.path.basename(d), rows))
    return out


def main():
    for arm in sys.argv[1:]:
        print("\n### %s" % arm)
        try:
            rs = runs(arm)
        except ImportError as e:
            print("  wandb not importable: %s" % e)
            continue
        if not rs:
            print("  no policy runs with an env/ success series")
            continue
        for i, (nm, rows) in enumerate(rs):
            if 0 not in rows:
                continue
            tag = "UNTRAINED BASELINE" if i == 0 else "warm-started from the previous round"
            vals = {k: v for k, v in rows[0].items() if "env/" in k}
            final = max((e for e in rows if e > 0), default=None)
            best = {}
            for k in vals:
                best[k] = max(v.get(k, float("-inf")) for v in rows.values())
            for k in sorted(vals):
                short = k.replace("env/", "")
                print("  %-26s %-34s epoch0 %.3f   best over the round %.3f   %s"
                      % (nm[-22:], short, vals[k], best[k], tag))
            if i == 0:
                print("  " + "-" * 92)
                print("  ^ READ: the epoch0 row above is an UNTRAINED policy. Compare it against")
                print("    every reported number for this arm. A reported value close to it means")
                print("    the metric is being satisfied by something other than learned skill.")
                print("  " + "-" * 92)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main())
