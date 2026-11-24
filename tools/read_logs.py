"""Three zero-compute reads from finished runs. No GPU, no environment, no model, no network.

  cem  <arm>...   does the CEM search need 30 opt steps? Reads "<prefix>/loss" and
                  "<prefix>/active", logged once per opt step by planning/cem.py:165-168, and
                  reports where the objective plateaus. Generation is ~63% of round wall-clock,
                  so this is the largest single lever on run time. (Audit M3b.)
  leak <wall_arm> -- <pusht_arm>...
                  size the pusht generation/scouting leak. goal25_train is leak-free for pusht
                  and goal25_valid is not; wall has no leak at all, so wall's
                  (valid - train) gap is pure split difficulty. Subtract it from pusht's gap.
                  (Audit X1.)
  exam <arm>...   how many DISTINCT tasks the reported metric actually contains, from
                  eval_episodes_*.csv. Also prints the first_success_step distribution when the
                  column is present, which is what decides eval_horizon. (Audit X2 and E6.)
"""
import csv, glob, json, os, re, sys
from collections import Counter


def runs(arm):
    """Every offline wandb run under an arm: (name, {step_or_epoch: {key: value}})."""
    from wandb.sdk.internal import datastore as wds
    from wandb.proto import wandb_internal_pb2 as pb
    for d in sorted(glob.glob(os.path.join(arm, "wandb", "*run-*"))):
        f = sorted(glob.glob(os.path.join(d, "*.wandb")))
        if not f:
            continue
        nm = os.path.basename(d)
        mf = os.path.join(d, "files", "wandb-metadata.json")
        if os.path.isfile(mf):
            try:
                nm = json.load(open(mf)).get("name") or nm
            except Exception:
                pass
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
                try:
                    g[k] = float(json.loads(it.value_json))
                except Exception:
                    pass
            ax = g.pop("step", g.pop("epoch", None))
            if ax is not None and g:
                rows.setdefault(int(ax), {}).update(g)
        yield nm, rows


def cmd_cem(arms):
    for arm in arms:
        print("\n### %s" % arm)
        any_run = False
        for nm, rows in runs(arm):
            keys = set(k for v in rows.values() for k in v)
            lk = sorted(k for k in keys if k.endswith("/loss"))
            ak = sorted(k for k in keys if k.endswith("/active"))
            if not lk:
                continue
            any_run = True
            for k in lk:
                pts = [(s, v[k]) for s, v in sorted(rows.items()) if k in v]
                if len(pts) < 4:
                    continue
                L = [p[1] for p in pts]
                lo, hi = min(L), L[0]
                span = hi - lo
                # first step within 5% and within 1% of the total improvement
                def reach(frac):
                    for i, v in enumerate(L):
                        if span <= 0 or (v - lo) <= frac * span:
                            return i + 1
                    return len(L)
                act = ""
                if ak:
                    A = [v[ak[0]] for s, v in sorted(rows.items()) if ak[0] in v]
                    if A:
                        act = "  active %g->%g" % (A[0], A[-1])
                print("  %-34s n=%-3d loss %.4g -> %.4g   95%% of gain by step %d, 99%% by %d%s"
                      % (nm[:34], len(L), hi, lo, reach(0.05), reach(0.01), act))
        if not any_run:
            print("  no run logged a '*/loss' series. Generation runs do; policy runs do not.")
    print("\nREAD: if 95% of the objective's improvement arrives well before step 30, the rest is")
    print("paid for nothing. Changing opt_steps is a change to the planner, so A/B it on")
    print("goal25_valid/max_success_rate before adopting -- but this says whether that A/B is")
    print("worth running at all.")


def _best_gaps(arm):
    """Per round: best goal25_valid and goal25_train, and the gap."""
    out = []
    for nm, rows in runs(arm):
        v = t = None
        for s in sorted(rows):
            for k, x in rows[s].items():
                if k.endswith("goal25_valid/max_success_rate"):
                    v = x if v is None else max(v, x)
                elif k.endswith("goal25_train/max_success_rate"):
                    t = x if t is None else max(t, x)
        if v is not None and t is not None:
            out.append((nm, v, t, v - t))
    return out


def cmd_leak(args):
    if "--" not in args:
        print("usage: leak <wall_arm>... -- <pusht_arm>...")
        return
    i = args.index("--")
    walls, pushts = args[:i], args[i + 1:]
    means = {}
    for label, arms in (("wall", walls), ("pusht", pushts)):
        gaps = []
        print("\n### %s" % label)
        for arm in arms:
            for nm, v, t, g in _best_gaps(arm):
                print("  %-40s valid %.3f  train %.3f  gap %+.3f" % (nm[:40], v, t, g))
                gaps.append(g)
        means[label] = (sum(gaps) / len(gaps)) if gaps else None
        print("  mean gap: %s over %d rounds"
              % ("n/a" if means[label] is None else "%+.3f" % means[label], len(gaps)))
    if means.get("wall") is not None and means.get("pusht") is not None:
        est = means["pusht"] - means["wall"]
        print("\n" + "=" * 70)
        print("pusht gap %+.3f  -  wall gap %+.3f  =  LEAK ESTIMATE %+.3f"
              % (means["pusht"], means["wall"], est))
        print("=" * 70)
        print("wall has no generation or scouting leak, so its gap is pure split difficulty.")
        print("Pusht's gap is that same difficulty PLUS the leak, so the difference estimates the")
        print("leak. This assumes the two environments' split-difficulty offsets are comparable,")
        print("which is the weak step -- treat it as an order of magnitude, not a point estimate.")


def cmd_exam(arms):
    for arm in arms:
        print("\n### %s" % arm)
        files = sorted(glob.glob(os.path.join(arm, "**", "eval_episodes_*.csv"), recursive=True))
        if not files:
            print("  no eval_episodes_*.csv found")
            continue
        print("  %d csv files (note: the path carries no epoch, so each eval OVERWROTE the"
              " previous one -- bug B3/X2)" % len(files))
        for p in files[-4:]:
            try:
                rows = list(csv.DictReader(open(p)))
            except Exception as e:
                print("   skip %s: %s" % (os.path.basename(p), e))
                continue
            if not rows:
                continue
            pairs = Counter((r.get("init_x"), r.get("init_y"), r.get("goal_x"), r.get("goal_y"))
                            for r in rows)
            inits = Counter((r.get("init_x"), r.get("init_y")) for r in rows)
            print("  %-46s episodes %d  distinct (init,goal) %d  distinct init %d"
                  % (os.path.basename(p)[:46], len(rows), len(pairs), len(inits)))
            top = pairs.most_common(3)
            print("       most repeated task drawn %s times; %d tasks drawn once"
                  % (", ".join(str(c) for _, c in top), sum(1 for c in pairs.values() if c == 1)))
            # How hard is this task set, and does it even require crossing the wall? A 'goal' pass
            # sets the goal to the state goal_offset frames AHEAD IN THE SAME TRAJECTORY
            # (train_policy.py:1297), so on wall -- whose demonstrations are random-action data --
            # the goal is wherever the random walk drifted to, not a deliberate crossing.
            try:
                gd = sorted(float(r["goal_dist"]) for r in rows if r.get("goal_dist"))
            except Exception:
                gd = []
            if gd:
                def q(f):
                    return gd[min(len(gd) - 1, int(f * len(gd)))]
                print("       goal_dist: median %.1f  p90 %.1f  max %.1f  (straight line, units)"
                      % (q(0.5), q(0.9), gd[-1]))
            try:
                WALL_X = 32.0
                cross = sum(1 for r in rows
                            if (float(r["init_x"]) - WALL_X) * (float(r["goal_x"]) - WALL_X) < 0)
                print("       episodes whose start and goal are on OPPOSITE sides of x=%g: "
                      "%d of %d (%.0f%%)" % (WALL_X, cross, len(rows), 100.0 * cross / len(rows)))
            except Exception:
                pass
            if "first_success_step" in rows[0]:
                hits = sorted(float(r["first_success_step"]) for r in rows
                              if float(r["first_success_step"]) >= 0)
                if hits:
                    def pc(q):
                        return hits[min(len(hits) - 1, int(q * len(hits)))]
                    print("       first_success_step over %d successes: median %.0f  p90 %.0f  "
                          "p99 %.0f  max %.0f"
                          % (len(hits), pc(0.5), pc(0.9), pc(0.99), hits[-1]))
                    print("       -> E6: eval_horizon above about %.0f buys nothing for"
                          " max_success_rate" % hits[-1])
            else:
                print("       no first_success_step column: this CSV predates the E5 change")


def cmd_horizon(arms):
    """E6 from data already on disk, for either environment.

    The relabel evaluation runs the policy on the planner's own tasks at EVERY horizon in
    relabel_eval_horizons (default 25 and 300) and writes them to relabel_eval.json. Comparing
    those two says directly how much the policy gains from 275 extra steps. If the long horizon
    adds nothing, eval_horizon: 300 is paying for steps that change no outcome.
    """
    for arm in arms:
        print("\n### %s" % arm)
        files = sorted(glob.glob(os.path.join(arm, "**", "relabel_eval.json"), recursive=True))
        if not files:
            print("  no relabel_eval.json found (needs relabel_eval_horizons and a relabel pass)")
            continue
        rows = []
        for p in files:
            try:
                js = json.load(open(p))
            except Exception as e:
                print("   skip %s: %s" % (p, e))
                continue
            for r in (js if isinstance(js, list) else [js]):
                hs = sorted(int(k.split("_h")[-1]) for k in r
                            if k.startswith("policy_max_success_h"))
                if len(hs) >= 2:
                    rows.append((os.path.basename(os.path.dirname(p)), hs,
                                 [r.get("policy_max_success_h%d" % h) for h in hs],
                                 r.get("planner_success_rate"), r.get("n_states")))
        if not rows:
            print("  found %d file(s) but none carry two horizons" % len(files))
            continue
        gains = []
        for nm, hs, vs, pl, n in rows:
            if None in vs:
                continue
            g = vs[-1] - vs[0]
            gains.append(g)
            print("  %-22s n=%-4s planner %-6s  policy h%d=%.3f  h%d=%.3f  gain from the long "
                  "horizon %+.3f" % (nm[:22], n, pl, hs[0], vs[0], hs[-1], vs[-1], g))
        if gains:
            m = sum(gains) / len(gains)
            print("\n  MEAN GAIN from the long horizon: %+.3f over %d rounds" % (m, len(gains)))
            if abs(m) < 0.02:
                print("  READ: the extra steps buy essentially nothing, so eval_horizon can be cut")
                print("  toward the short horizon. Confirm with first_success_step once a round has")
                print("  run with the E5 change.")
            else:
                print("  READ: the long horizon does add success, so the policy is SLOW rather than")
                print("  incapable on these tasks, and eval_horizon should NOT be cut to the short")
                print("  horizon. first_success_step will say where it can be cut to.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "horizon":
        cmd_horizon(args)
    elif cmd == "cem":
        cmd_cem(args)
    elif cmd == "leak":
        cmd_leak(args)
    elif cmd == "exam":
        cmd_exam(args)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
