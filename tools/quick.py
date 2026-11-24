"""Two reads that need no wandb at all. Pure standard library, works in any env.

  cross <arm>...    M11: is the goal25 task actually a cross-wall task? Reads
                    eval_episodes_*.csv and prints goal_dist plus the fraction of episodes whose
                    start and goal lie on opposite sides of the wall. Also prints
                    first_success_step when present, which is E6's answer.
  horizon <arm>...  E6: what do the extra steps buy? Reads relabel_eval.json, which already holds
                    the policy's success at every relabel_eval_horizons value.
"""
import csv, glob, json, os, sys

WALL_X = 32.0


def cmd_cross(arms):
    for arm in arms:
        print("\n### %s" % arm)
        seen = set()
        for p in sorted(glob.glob(os.path.join(arm, "**", "eval_episodes_*.csv"), recursive=True)):
            b = os.path.basename(p)
            if b in seen:
                continue
            seen.add(b)
            try:
                rows = list(csv.DictReader(open(p)))
            except Exception as e:
                print("  skip %s: %s" % (b, e)); continue
            if not rows:
                continue
            gd = sorted(float(r["goal_dist"]) for r in rows if r.get("goal_dist"))
            cross = sum(1 for r in rows
                        if (float(r["init_x"]) - WALL_X) * (float(r["goal_x"]) - WALL_X) < 0)
            def q(f):
                return gd[min(len(gd) - 1, int(f * len(gd)))] if gd else float("nan")
            print("  %-34s n=%-4d goal_dist median %5.1f p90 %5.1f max %5.1f   "
                  "start/goal on OPPOSITE sides of x=%g: %d (%.0f%%)"
                  % (b[:34], len(rows), q(0.5), q(0.9), gd[-1] if gd else float("nan"),
                     WALL_X, cross, 100.0 * cross / len(rows)))
            if rows and "first_success_step" in rows[0]:
                h = sorted(float(r["first_success_step"]) for r in rows
                           if float(r["first_success_step"]) >= 0)
                if h:
                    print("       first_success_step: median %.0f p90 %.0f max %.0f "
                          "-> eval_horizon above %.0f changes nothing"
                          % (h[len(h) // 2], h[min(len(h) - 1, int(0.9 * len(h)))], h[-1], h[-1]))
    print("\nREAD (wall only; x=32 is meaningless for pusht): a low crossing percentage means the")
    print("goal25 passes were not testing cross-wall traversal, which is M11.")


def cmd_horizon(arms):
    for arm in arms:
        print("\n### %s" % arm)
        files = sorted(glob.glob(os.path.join(arm, "**", "relabel_eval.json"), recursive=True))
        if not files:
            print("  no relabel_eval.json found"); continue
        gains = []
        for p in files:
            try:
                js = json.load(open(p))
            except Exception as e:
                print("  skip %s: %s" % (p, e)); continue
            for r in (js if isinstance(js, list) else [js]):
                hs = sorted(int(k.split("_h")[-1]) for k in r
                            if k.startswith("policy_max_success_h"))
                vs = [r.get("policy_max_success_h%d" % h) for h in hs]
                if len(hs) < 2 or None in vs:
                    continue
                gains.append(vs[-1] - vs[0])
                print("  %-24s planner %-6s policy h%d=%.3f h%d=%.3f  long-horizon gain %+.3f"
                      % (os.path.basename(os.path.dirname(p))[:24],
                         r.get("planner_success_rate"), hs[0], vs[0], hs[-1], vs[-1], gains[-1]))
        if gains:
            m = sum(gains) / len(gains)
            print("  MEAN long-horizon gain %+.3f over %d rounds" % (m, len(gains)))
            print("  READ: %s" % ("near zero, so eval_horizon can be cut toward the short horizon"
                                  if abs(m) < 0.02 else
                                  "real, so the policy is SLOW rather than incapable; do not cut "
                                  "to the short horizon"))


def cmd_detour(arms):
    """Is a success a deliberate solve or an accidental brush past the goal?

    max_success_rate credits an episode that EVER came within the success radius, so an agent
    wandering for 300 steps can collect credit by chance. path_len is the distance actually
    travelled and goal_dist the straight-line distance to the goal, so path_len / goal_dist is a
    detour ratio: near 1 means it went more or less straight there, large means it wandered.
    """
    for arm in arms:
        print("\n### %s" % arm)
        seen = set()
        for p in sorted(glob.glob(os.path.join(arm, "**", "eval_episodes_*.csv"), recursive=True)):
            b = os.path.basename(p)
            if b in seen:
                continue
            seen.add(b)
            try:
                rows = list(csv.DictReader(open(p)))
            except Exception as e:
                print("  skip %s: %s" % (b, e)); continue
            if not rows or "path_len" not in rows[0]:
                print("  %-34s no path_len column" % b[:34]); continue
            def num(r, k):
                try:
                    return float(r[k])
                except Exception:
                    return float("nan")
            succ = [r for r in rows if num(r, "success_max") >= 0.5]
            fail = [r for r in rows if num(r, "success_max") < 0.5]
            def ratios(rs):
                out = []
                for r in rs:
                    g, pl = num(r, "goal_dist"), num(r, "path_len")
                    if g > 1e-6 and pl == pl:
                        out.append(pl / g)
                return sorted(out)
            rs, rf = ratios(succ), ratios(fail)
            def med(v):
                return v[len(v) // 2] if v else float("nan")
            print("  %-34s successes %3d  detour ratio median %6.1f   failures %3d  median %6.1f"
                  % (b[:34], len(succ), med(rs), len(fail), med(rf)))
            if rs:
                print("       successes: path travelled / straight-line distance -- p10 %.1f  "
                      "p90 %.1f  max %.1f" % (rs[max(0, int(0.1 * len(rs)) - 1)],
                                              rs[min(len(rs) - 1, int(0.9 * len(rs)))], rs[-1]))
            cd = sorted(num(r, "closest_dist") for r in fail
                        if num(r, "closest_dist") == num(r, "closest_dist"))
            if cd:
                print("       failures: closest approach median %.1f  p10 %.1f (success radius is "
                      "about 4.5 on wall)" % (med(cd), cd[max(0, int(0.1 * len(cd)) - 1)]))
    print("\nREAD: a detour ratio near 1-2 on successes means the policy goes to the goal, so the")
    print("extra horizon is buying real but SLOW solving. A large ratio, especially one much larger")
    print("than on failures, means successes are long wanders that happened to pass through the")
    print("goal region, and the long-horizon gain is mostly luck.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    c, a = sys.argv[1], sys.argv[2:]
    if c == "cross":
        cmd_cross(a)
    elif c == "horizon":
        cmd_horizon(a)
    elif c == "detour":
        cmd_detour(a)
    else:
        print(__doc__); sys.exit(1)
