"""Consensus gating: turn per-voter votes into silver labels + confidence tiers.

Each voter emits a vote in {PZ, NG, ABSTAIN}.
Gates (n_pz, n_ng = non-abstaining votes for each pole):
  default (majority) : one pole >= 3 votes AND the other <= 1 dissent
  strict             : one pole >= 4 votes AND 0 dissent          (== "unanimous")
  extended (lenient) : majority-labeled  OR  exactly two voters spoke and both agree (2-0)

"""
from __future__ import annotations
from dataclasses import dataclass

PZ, NG, ABSTAIN = "PZ", "NG", "ABSTAIN"


@dataclass
class SegmentVotes:
    seg_id: str
    votes: list[str]          # one of PZ/NG/ABSTAIN per voter
    speaker_role: str = "child"

    @property
    def n_pz(self) -> int: return sum(v == PZ for v in self.votes)
    @property
    def n_ng(self) -> int: return sum(v == NG for v in self.votes)
    @property
    def n_real(self) -> int: return self.n_pz + self.n_ng


def label_default(s: SegmentVotes) -> str | None:
    """Majority gate: >=3 for one pole, <=1 dissent."""
    if s.n_pz >= 3 and s.n_ng <= 1:
        return PZ
    if s.n_ng >= 3 and s.n_pz <= 1:
        return NG
    return None


def label_strict(s: SegmentVotes) -> str | None:
    """Unanimous gate: >=4 for one pole, 0 dissent."""
    if s.n_pz >= 4 and s.n_ng == 0:
        return PZ
    if s.n_ng >= 4 and s.n_pz == 0:
        return NG
    return None


def label_extended(s: SegmentVotes) -> str | None:
    """Extended gate: majority OR a clean 2-0 agreement."""
    lab = label_default(s)
    if lab is not None:
        return lab
    if s.n_real == 2 and (s.n_pz == 0 or s.n_ng == 0):
        return PZ if s.n_pz == 2 else NG
    return None


def confidence_weight(s: SegmentVotes) -> float:
    return 1.0 if s.n_real >= 3 else 0.5


def outcome(s: SegmentVotes, preset: str = "extended") -> tuple[str | None, str]:
    """Return (label, reason). reason in {labeled, insufficient_votes, low_agreement}."""
    gate = {"default": label_default, "strict": label_strict, "extended": label_extended}[preset]
    lab = gate(s)
    if lab is not None:
        return lab, "labeled"
    return None, ("insufficient_votes" if s.n_real < 2 else "low_agreement")


if __name__ == "__main__":
    # sanity: reproduce the tier sizes from the paper on a votes CSV
    import argparse, csv, collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes_csv", required=True, help="cols: seg_id, v1..v5 in {PZ,NG,ABSTAIN}, role")
    args = ap.parse_args()
    counts = {p: collections.Counter() for p in ("strict", "default", "extended")}
    pz = {p: 0 for p in counts}
    with open(args.votes_csv) as f:
        for r in csv.DictReader(f):
            votes = [r[f"v{i}"] for i in range(1, 6)]
            s = SegmentVotes(r["seg_id"], votes, r.get("role", "child"))
            for p in counts:
                lab, why = outcome(s, p)
                counts[p]["labeled" if lab else why] += 1
                pz[p] += (lab == PZ)
    for p in ("strict", "default", "extended"):
        n = counts[p]["labeled"]
        print(f"{p:9s} labeled={n:5d}  PZ={pz[p]/n*100:.1f}%  {dict(counts[p])}")
