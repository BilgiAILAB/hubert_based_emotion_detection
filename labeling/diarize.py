import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_ID     = "medkit/simsamu-diarization"
SAMPLE_RATE  = 16_000   # output sample rate for all clips
MIN_DURATION = 1.0      # seconds — clips shorter than this are dropped
MAX_DURATION = 30.0     # seconds — clips longer than this are split at silence
MERGE_GAP    = 0.4      # seconds — same-speaker segments closer than this are merged


# helpers 

def load_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as float32 numpy array resampled to SAMPLE_RATE."""
    import librosa
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    return y, sr


def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply stationary noise reduction."""
    return nr.reduce_noise(y=audio, sr=sr, stationary=True)


def save_clip(audio: np.ndarray, sr: int, out_path: Path):
    """Save a numpy audio array as 16-bit WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr, subtype="PCM_16")


# Speaker summary 

def print_speaker_summary(diarization):
    """Print total speech time per speaker (informational only)."""
    speaker_duration: dict[str, float] = {}
    speaker_turns: dict[str, int] = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        speaker_duration[speaker] = speaker_duration.get(speaker, 0.0) + dur
        speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1

    print(f"\n  Speaker totals (all will be exported):")
    for sp, dur in sorted(speaker_duration.items(), key=lambda x: -x[1]):
        print(f"    {sp}: {dur:.1f}s  ({speaker_turns[sp]} turns)")


# Segment merging and splitting 

def merge_close_segments(
    segments: list[tuple[float, float]], gap: float
) -> list[tuple[float, float]]:
    """Merge consecutive segments whose gap is less than `gap` seconds."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def split_long_segment(
    audio: np.ndarray, sr: int,
    start_sec: float, end_sec: float,
    max_dur: float,
) -> list[tuple[float, float]]:
    """
    Recursively split a segment that exceeds max_dur at the quietest
    point in its middle third. Returns a list of (start, end) pairs.
    """
    if end_sec - start_sec <= max_dur:
        return [(start_sec, end_sec)]

    import librosa
    s = int(start_sec * sr)
    e = int(end_sec * sr)
    clip = audio[s:e]
    frame_len = int(0.01 * sr)   # 10ms frames
    rms = librosa.feature.rms(y=clip, frame_length=frame_len, hop_length=frame_len)[0]

    mid_start = len(rms) // 3
    mid_end   = 2 * len(rms) // 3
    split_frame = mid_start + int(np.argmin(rms[mid_start:mid_end]))
    split_sec = start_sec + split_frame * 0.01

    return (
        split_long_segment(audio, sr, start_sec, split_sec, max_dur) +
        split_long_segment(audio, sr, split_sec, end_sec,   max_dur)
    )


# Main diarization logic 

def diarize_session(
    wav_path: Path,
    out_dir: Path,
    pipeline,
    audio: np.ndarray,
    sr: int,
    num_speakers: int,
    min_dur: float,
    max_dur: float,
) -> int:
    """Diarize one session and export all speakers' clips. Returns clip count."""
    session_id = wav_path.stem
    print(f"\n[*] Diarizing: {wav_path.name}")

    with torch.no_grad():
        result = pipeline(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr},
            num_speakers=num_speakers,
        )

    # Normalise pyannote output — different versions return different types
    if hasattr(result, "itertracks"):
        diarization = result                         
    elif hasattr(result, "speaker_diarization"):
        diarization = result.speaker_diarization      
    elif hasattr(result, "diarization"):
        diarization = result.diarization
    else:
        attrs = [a for a in dir(result) if not a.startswith("_")]
        raise RuntimeError(f"Unrecognised pyannote output: {type(result)}, attrs: {attrs}")

    print_speaker_summary(diarization)

    # Group turns by speaker
    speaker_segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments[speaker].append((turn.start, turn.end))

    exported = 0
    total_raw = 0
    for speaker, raw_segments in speaker_segments.items():
        raw_segments.sort(key=lambda x: x[0])
        total_raw += len(raw_segments)

        merged = merge_close_segments(raw_segments, gap=MERGE_GAP)

        final_segments: list[tuple[float, float]] = []
        for start, end in merged:
            final_segments.extend(split_long_segment(audio, sr, start, end, max_dur))

        for idx, (start, end) in enumerate(final_segments):
            if end - start < min_dur:
                continue
            clip = audio[int(start * sr):int(end * sr)]
            clip_name = f"{session_id}_{speaker}_seg{idx:04d}.wav"
            save_clip(clip, sr, out_dir / clip_name)
            exported += 1

    print(f"  → Exported {exported} clips from all speakers  "
          f"(from {total_raw} raw turns)")
    return exported


# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="Diarize therapy sessions and export all speaker utterances as WAV clips."
    )
    parser.add_argument("--sessions_dir", default="data/sessions",
                        help="Folder containing full session WAV recordings.")
    parser.add_argument("--out_dir",      default="data/segments",
                        help="Output folder for utterance clips.")
    parser.add_argument("--hf_token",     default=None,
                        help="HuggingFace token (or set HF_TOKEN env var). "
                             "Required for medkit/simsamu-diarization.")
    parser.add_argument("--num_speakers", default=2, type=int,
                        help="Expected number of speakers per session (default: 3).")
    parser.add_argument("--min_duration", default=MIN_DURATION, type=float,
                        help=f"Minimum clip duration in seconds (default: {MIN_DURATION}).")
    parser.add_argument("--max_duration", default=MAX_DURATION, type=float,
                        help=f"Maximum clip duration in seconds (default: {MAX_DURATION}).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sessions that already have segments in the output directory.")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "\n[!] No HuggingFace token provided.\n"
        )
        raise SystemExit(1)

    session_files = sorted(sessions_dir.glob("*.wav"))
    if not session_files:
        print(f"[!] No WAV files found in {sessions_dir}")
        raise SystemExit(1)

    if args.skip_existing:
        existing_prefixes = {
            "_".join(f.stem.rsplit("_", 3)[:-3])
            for f in out_dir.glob("*.wav")
        }
        before = len(session_files)
        session_files = [f for f in session_files if f.stem not in existing_prefixes]
        skipped = before - len(session_files)
        if skipped:
            print(f"[*] Skipping {skipped} already-diarized sessions")

    print(f"[*] Found {len(session_files)} session recordings to process")

    print(f"[*] Loading {MODEL_ID} ...")
    from pyannote.audio import Pipeline
    diar_pipeline = Pipeline.from_pretrained(MODEL_ID, token=hf_token)

    if torch.cuda.is_available():
        diar_pipeline = diar_pipeline.to(torch.device("cuda"))
        print("[+] Diarization running on CUDA GPU")
    elif torch.backends.mps.is_available():
        diar_pipeline = diar_pipeline.to(torch.device("mps"))
        print("[+] Diarization running on Apple MPS (Metal)")
    else:
        print("[*] Diarization running on CPU")

    total_clips = 0
    for session_path in tqdm(session_files, desc="Sessions", unit="session"):
        print(f"[*] Loading audio ({session_path.stat().st_size / 1e6:.0f} MB)...")
        audio, sr = load_audio(session_path)
        print(f"    Duration: {len(audio)/sr/60:.1f} minutes")
        print(f"    Applying noise reduction...")
        audio = denoise(audio, sr)

        n = diarize_session(
            wav_path=session_path,
            out_dir=out_dir,
            pipeline=diar_pipeline,
            audio=audio,
            sr=sr,
            num_speakers=args.num_speakers,
            min_dur=args.min_duration,
            max_dur=args.max_duration,
        )
        total_clips += n

    print(f"\n{'='*50}")
    print(f"  DIARIZATION COMPLETE")
    print(f"  Sessions processed : {len(session_files)}")
    print(f"  Clips exported     : {total_clips}")
    print(f"  Output directory   : {out_dir}")
    print(f"\n  Next step:")
    print(f"  python label_pipeline.py --segments_dir {out_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
