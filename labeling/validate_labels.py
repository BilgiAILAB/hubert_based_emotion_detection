"""
Label-free validation of the silver labels.

Confronts the consensus labels with evidence channels the majority vote did NOT use:
  1. laughter clips should be PZ (>=70%)          [event detector, not the acoustic vote]
  2. positive-keyword-only clips should be PZ (>=60%) and negative-keyword-only NG (>=60%)
  3. session-normalized energy(RMS) higher for PZ (one-sided Mann-Whitney, p<.05)
  4. session-normalized pitch(F0) higher for PZ (sanity; known weak for child distress)
  5. leave-one-voter-out re-labeling flips < 5%; report coverage retention per voter

"""
from __future__ import annotations
import numpy as np
from scipy.stats import mannwhitneyu


def concordance(labels, mask, target):
    sel = [lab for lab, m in zip(labels, mask) if m]
    return (sum(l == target for l in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)


def mannwhitney_pz_gt_ng(values, labels):
    pz = [v for v, l in zip(values, labels) if l == "PZ"]
    ng = [v for v, l in zip(values, labels) if l == "NG"]
    if not pz or not ng:
        return float("nan")
    return mannwhitneyu(pz, ng, alternative="greater").pvalue


def leave_one_voter_out(seg_votes, relabel_fn):
    """For each voter, drop it, re-derive labels, report (flip_rate, coverage_retention)."""
    base = {s.seg_id: relabel_fn(s.votes) for s in seg_votes}
    base_lab = {k: v for k, v in base.items() if v is not None}
    out = {}
    n_voters = len(seg_votes[0].votes)
    for j in range(n_voters):
        flips = kept = 0
        for s in seg_votes:
            reduced = [v for i, v in enumerate(s.votes) if i != j]
            new = relabel_fn(reduced)
            if s.seg_id in base_lab and new is not None:
                kept += 1
                flips += (new != base_lab[s.seg_id])
        out[j] = (flips / max(kept, 1), kept / max(len(base_lab), 1))
    return out


def report(labels, laughter_mask, poskw_mask, negkw_mask, rms, f0):
    lines = ["============== VALIDATION REPORT (label-free) ================\n"]
    c, n = concordance(labels, laughter_mask, "PZ")
    lines.append(f"{'PASS' if c>=.70 else 'WARN'}  laughter clips labeled PZ: {c*100:.1f}% (n={n}, needs >=70%)")
    c, n = concordance(labels, poskw_mask, "PZ")
    lines.append(f"{'PASS' if c>=.60 else 'WARN'}  positive-keyword clips labeled PZ: {c*100:.1f}% (n={n}, needs >=60%)")
    c, n = concordance(labels, negkw_mask, "NG")
    lines.append(f"{'PASS' if c>=.60 else 'WARN'}  negative-keyword clips labeled NG: {c*100:.1f}% (n={n}, needs >=60%)")
    p = mannwhitney_pz_gt_ng(f0, labels)
    lines.append(f"{'PASS' if p<.05 else 'WARN'}  pitch(F0) PZ>NG (Mann-Whitney p={p:.3g})")
    p = mannwhitney_pz_gt_ng(rms, labels)
    lines.append(f"{'PASS' if p<.05 else 'WARN'}  energy(RMS) PZ>NG (Mann-Whitney p={p:.3g})")
    return "\n".join(lines)
