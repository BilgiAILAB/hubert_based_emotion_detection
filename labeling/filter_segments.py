import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

MIN_DURATION    = 5.0    
MAX_DURATION    = 25.0   
MIN_RMS_ENERGY  = 0.002  # clips below this are considered near-silent
MAX_PER_SESSION = 180    # cap per session 
MIN_PER_SESSION = 20     # floor per session 
SAMPLE_RATE     = 16_000
RANDOM_SEED     = 42

def clip_stats(wav_path: Path) -> dict:
    """Return duration and mean RMS energy for a WAV file."""
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    return {"duration": duration, "rms": rms}


def session_id(wav_path: Path) -> str:
    """
    Extract session identifier from filename.
    """
    name = wav_path.stem
    parts = name.rsplit("_", 3)   
    if len(parts) >= 4:
        return "_".join(parts[:-3])
    return name 


def main():
    parser = argparse.ArgumentParser(
        description="Filter segments before LLM labeling."
    )
    parser.add_argument("--in_dir",          default="data/segments")
    parser.add_argument("--out_dir",         default="data/segments_filtered")
    parser.add_argument("--min_dur",         default=MIN_DURATION,    type=float)
    parser.add_argument("--max_dur",         default=MAX_DURATION,    type=float)
    parser.add_argument("--min_energy",      default=MIN_RMS_ENERGY,  type=float)
    parser.add_argument("--max_per_session", default=MAX_PER_SESSION, type=int)
    parser.add_argument("--min_per_session", default=MIN_PER_SESSION, type=int)
    parser.add_argument("--dry_run",         action="store_true",
                        help="Print stats only — do not copy files.")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    random.seed(RANDOM_SEED)

    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        print(f"[!] No WAV files found in {in_dir}")
        return

    print(f"[*] Found {len(wavs)} segments in {in_dir}")
    print(f"[*] Computing duration and energy for each clip...")

    # compute stats 
    records = []
    for wav in tqdm(wavs, unit="clip"):
        try:
            stats = clip_stats(wav)
            records.append({
                "path":     wav,
                "session":  session_id(wav),
                "duration": stats["duration"],
                "rms":      stats["rms"],
            })
        except Exception as e:
            print(f"  [!] Skipping {wav.name}: {e}")

    total = len(records)

    # duration filter 
    after_dur = [r for r in records
                 if args.min_dur <= r["duration"] <= args.max_dur]
    print(f"\n── Duration filter ({args.min_dur}–{args.max_dur}s) ─────────────")
    print(f"  Before: {total}  →  After: {len(after_dur)}  "
          f"(removed {total - len(after_dur)})")

    # energy filter 
    after_energy = [r for r in after_dur if r["rms"] >= args.min_energy]
    print(f"\n── Energy filter (RMS ≥ {args.min_energy}) ─────────────────────")
    print(f"  Before: {len(after_dur)}  →  After: {len(after_energy)}  "
          f"(removed {len(after_dur) - len(after_energy)})")

    # per-session cap / floor
    by_session: dict[str, list] = defaultdict(list)
    for r in after_energy:
        by_session[r["session"]].append(r)

    print(f"\n── Per-session cap (max={args.max_per_session}, "
          f"min={args.min_per_session}) ──────────")

    final = []
    sessions_below_floor = []
    for sess, clips in sorted(by_session.items()):
        original_count = len(clips)

        if original_count > args.max_per_session:
            clips = random.sample(clips, args.max_per_session)

        if len(clips) < args.min_per_session:
            sessions_below_floor.append((sess, len(clips)))

        final.extend(clips)
        print(f"  {sess[:55]:<55}  {original_count:>5} → {len(clips)}")

    if sessions_below_floor:
        print(f"\n  ⚠️  Sessions with fewer than {args.min_per_session} clips "
              f"after filtering:")
        for sess, n in sessions_below_floor:
            print(f"     {sess}: {n} clips  (kept all)")

    # Summary 
    print(f"\n{'='*55}")
    print(f"  Total segments before : {total}")
    print(f"  After duration filter : {len(after_dur)}")
    print(f"  After energy filter   : {len(after_energy)}")
    print(f"  After session cap     : {len(final)}")
    print(f"  Sessions retained     : {len(by_session)}")
    print(f"{'='*55}")

    if args.dry_run:
        print("\n  Dry run — no files copied.")
        return

    # Copy surviving clips
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] Copying {len(final)} clips to {out_dir} ...")
    for r in tqdm(final, unit="clip"):
        shutil.copy2(r["path"], out_dir / r["path"].name)

    print(f"\n[+] Done. {len(final)} segments ready in {out_dir}")
    print(f"\n  Next step:")
    print(f"  python label_pipeline.py --segments_dir {out_dir}")


if __name__ == "__main__":
    main()
