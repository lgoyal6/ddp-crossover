"""Step 5 gate: the crossover, the all-reduce fraction curve, and what each
mitigation recovers."""
from __future__ import annotations

import argparse
import json
import os


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("rows", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/ddp_sweep.json")
    ap.add_argument("--nccl", default="results/nccl_fp16.json")
    a = ap.parse_args()

    rows = load(a.sweep)
    if not rows:
        print(f"no rows in {a.sweep}; run run_all.sh first")
        return

    def pick(tag, **kw):
        out = [r for r in rows if r["tag"] == tag
               and all(r.get(k) == v for k, v in kw.items())]
        return {r["width"]: r for r in out}

    one = pick("1gpu")
    two = pick("2gpu_ddp", bucket_mb=25, accum=1)

    print("=== crossover: does the second GPU pay? ===")
    print(f"{'width':>7}{'params_M':>10}{'grad_MiB':>10}{'1gpu sps':>11}"
          f"{'2gpu sps':>11}{'scaling':>9}{'comm%':>8}  verdict")
    crossover = None
    for w in sorted(set(one) & set(two)):
        o, t = one[w], two[w]
        sc = t["samples_per_s"] / o["samples_per_s"]
        v = ("2 GPUs SLOWER" if sc < 1.0
             else "sublinear" if sc < 1.6 else "scales")
        if sc < 1.0:
            crossover = w
        print(f"{w:>7}{t['params']/1e6:>10.2f}{t['grad_mib']:>10.1f}"
              f"{o['samples_per_s']:>11.0f}{t['samples_per_s']:>11.0f}"
              f"{sc:>9.2f}{100*t['comm_frac']:>7.1f}%  {v}")

    # The crossover is not assumed to run in either direction. The original
    # hypothesis was "small models lose"; the measurement says the opposite, so
    # this reports the boundary it actually finds rather than the one expected.
    widths = sorted(set(one) & set(two))
    scal = {w: two[w]["samples_per_s"] / one[w]["samples_per_s"] for w in widths}
    wins = [w for w in widths if scal[w] >= 1.0]
    loses = [w for w in widths if scal[w] < 1.0]
    if wins and loses:
        if max(wins) < min(loses):
            print(f"\nCROSSOVER: the second GPU pays up to width {max(wins)} "
                  f"({two[max(wins)]['params']/1e6:.2f}M params, "
                  f"{two[max(wins)]['grad_mib']:.1f} MiB of gradient) and "
                  f"stops paying from width {min(loses)} "
                  f"({two[min(loses)]['params']/1e6:.2f}M params) upward.")
            print("  Note this runs OPPOSITE to the usual expectation that "
                  "small models are the ones that fail to scale.")
        else:
            print(f"\nNon-monotonic: wins at {wins}, loses at {loses}.")
    elif loses:
        print(f"\nThe second GPU never paid at any width in this sweep "
              f"({min(loses)}..{max(loses)}); extend --widths downward.")
    else:
        print("\nNo crossover: 2 GPUs were never slower. Extend --widths upward.")

    nccl = load(a.nccl)
    if nccl:
        peak = max(r["allreduce_busbw_gbs"] for r in nccl)
        floor = nccl[0]["allreduce_s"] * 1e6
        print(f"\nNCCL context: peak all-reduce bus bandwidth {peak:.2f} GB/s, "
              f"small-message floor {floor:.1f} us")
        print("A gradient buffer that is small enough to sit on the latency "
              "floor pays the same cost no matter how small it gets, which is "
              "what makes the second GPU a net loss below the crossover.")

    print("\n=== what each mitigation recovers ===")
    base = two
    for tag, label, key in (("bucket", "bucket_cap_mb", "bucket_mb"),
                            ("accum", "grad accumulation", "accum"),
                            ("fp16hook", "fp16 grad compression", None),
                            ("fsdp", "FSDP instead of DDP", None)):
        got = [r for r in rows if r["tag"] == tag]
        if not got:
            continue
        print(f"\n  {label}")
        vals = sorted({r[key] for r in got}) if key else [None]
        print(f"{'width':>9}" + "".join(
            f"{(str(v) if v is not None else tag):>12}" for v in vals)
            + f"{'baseline':>12}")
        for w in sorted({r["width"] for r in got}):
            line = f"{w:>9}"
            for v in vals:
                m = [r for r in got if r["width"] == w
                     and (key is None or r[key] == v)]
                line += f"{(m[0]['samples_per_s'] if m else float('nan')):>12.0f}"
            b = base.get(w)
            line += f"{(b['samples_per_s'] if b else float('nan')):>12.0f}"
            print(line)


if __name__ == "__main__":
    main()
