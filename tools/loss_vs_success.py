import glob, json, math, os, re, sys
# match by SUBSTRING, so a logging prefix change cannot break this again.
SUCPAT = "max_success_rate"
LOSSPAT = ("epoch_wise_action_diff", "exec_mse")
PREF_SUC = ["env/goal25_valid/max_success_rate", "goal25_valid/max_success_rate"]
PREF_LOSS = ["eval/epoch_wise_action_diff", "offline/all/exec_mse",
             "eval/epoch_wise_action_diff_tot"]

def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0]*len(x); i = 0
    while i < len(o):
        j = i
        while j+1 < len(o) and x[o[j+1]] == x[o[i]]: j += 1
        a = (i+j)/2.0+1
        for k in range(i, j+1): r[o[k]] = a
        i = j+1
    return r

def pear(x, y):
    n = len(x)
    if n < 3: return float("nan")
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    return float("nan") if sxx <= 0 or syy <= 0 else sxy/math.sqrt(sxx*syy)

def spear(x, y): return pear(ranks(x), ranks(y))

def one_run(d):
    from wandb.sdk.internal import datastore as wds
    from wandb.proto import wandb_internal_pb2 as pb
    f = sorted(glob.glob(os.path.join(d, "*.wandb")))
    if not f: return None, {}
    nm = os.path.basename(d)
    mf = os.path.join(d, "files", "wandb-metadata.json")
    if os.path.isfile(mf):
        try: nm = json.load(open(mf)).get("name") or nm
        except Exception: pass
    rows = {}; ds = wds.DataStore(); ds.open_for_scan(f[0])
    while True:
        try: data = ds.scan_data()
        except Exception: break
        if data is None: break
        rec = pb.Record()
        try: rec.ParseFromString(data)
        except Exception: continue
        if not rec.HasField("history"): continue
        g = {}
        for it in rec.history.item:
            k = it.key or ".".join(it.nested_key)
            if k == "epoch" or SUCPAT in k or any(p in k for p in LOSSPAT):
                try: g[k] = float(json.loads(it.value_json))
                except Exception: pass
        if "epoch" in g and len(g) > 1:
            rows.setdefault(int(g.pop("epoch")), {}).update(g)
    return nm, rows

def from_wandb(arm):
    out = {}; bad = 0
    for d in sorted(glob.glob(os.path.join(arm, "wandb", "*run-*"))):
        if not os.path.isdir(d): continue
        try:
            nm, rows = one_run(d)            # per-directory: one corrupt run cannot kill the arm
        except Exception as e:
            bad += 1
            if bad <= 3: print("   skip %s: %s" % (os.path.basename(d), e))
            continue
        if not rows: continue
        if not any(any(p in k for p in LOSSPAT) for v in rows.values() for k in v):
            continue                          # not a policy-training run
        m = re.search(r"r(\d+)", nm or "")
        lab = "r%02d" % int(m.group(1)) if m else (nm or os.path.basename(d))
        if lab in out: out[lab].update(rows)  # a resumed round writes two run dirs
        else: out[lab] = rows
    if bad: print("   (%d run dirs unreadable, skipped)" % bad)
    return out

def pick(keys, pref, pats):
    for p in pref:
        if p in keys: return p
    c = sorted(k for k in keys if any(p in k for p in pats))
    return c[0] if c else None

def main():
    for arm in sys.argv[1:]:
        print("\n### %s" % arm)
        try: rows = from_wandb(arm)
        except ImportError as e:
            print("  wandb not importable: %s  (conda activate dino_wm)" % e); continue
        if not rows:
            print("  no policy runs with a loss series found"); continue
        keys = set(k for r in rows.values() for v in r.values() for k in v)
        print("  keys found: %s" % sorted(keys))
        lk = pick(keys, PREF_LOSS, LOSSPAT)
        sk = pick(keys, PREF_SUC, (SUCPAT,))
        if lk is None or sk is None:
            print("  cannot pair: loss=%r success=%r" % (lk, sk)); continue
        print("  loss=%s   success=%s" % (lk, sk))
        res = []
        for lab in sorted(rows):
            p = [(e, v[lk], v[sk]) for e, v in sorted(rows[lab].items()) if lk in v and sk in v]
            if len(p) < 3:
                print("  %-5s n=%d too few paired points" % (lab, len(p))); continue
            ep = [q[0] for q in p]; L = [q[1] for q in p]; S = [q[2] for q in p]
            il = min(range(len(p)), key=lambda i: L[i]); isc = max(range(len(p)), key=lambda i: S[i])
            r = {"rho": spear(L, S), "reg": S[isc]-S[il], "el": ep[il], "es": ep[isc]}
            res.append(r)
            print("  %-5s n=%-3d rho %+0.3f  lossmin@%-5d succ %.3f | best %.3f @%-5d | regret %+.3f"
                  % (lab, len(p), r["rho"], r["el"], S[il], S[isc], r["es"], r["reg"]))
        rr = [x["rho"] for x in res if x["rho"] == x["rho"]]
        if rr:
            print("  POOLED rounds=%d  mean rho %+0.3f (negative=good)  mean regret %+0.3f  "
                  "loss bottoms early %d/%d"
                  % (len(res), sum(rr)/len(rr), sum(x["reg"] for x in res)/len(res),
                     sum(1 for x in res if x["el"] < x["es"]), len(res)))

main()
