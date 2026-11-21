"""把讲解/实战视频转成「可读的训练材料」：语音转写 + 关键帧。

为什么要自己写：`dsh-video-understanding` 的视觉后端依赖 CUDA，本机只有集显；
它的语音后端就是 faster-whisper small（CPU INT8），这里直接用同一套权重，
再用 ffmpeg 抽帧让决策模型自己看图（帧是画面文字/数字的唯一事实来源）。

产物（默认落在 ``training/artifacts/video/<视频名>/``）::

    audio.f32            16kHz 单声道原始浮点 PCM（可选，--keep-audio）
    asr.json             {"segments":[{"start","end","text"}], "duration", "range"}
    asr.txt              [mm:ss] 文本，便于人读
    frames/f-000123-0123.5s.jpg   按 --frame-every 抽的关键帧
    notes.json           本次运行的参数与统计

用法::

    python scripts/video_notes.py "D:/downloads/traingVideos/0.mp4" --start 00:41 --end 02:30
    python scripts/video_notes.py 3.mp4 --frame-every 12 --no-asr

时间轴以**原视频**为准，`--start/--end` 只裁切，不重置时间戳。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(r"D:\_tool\whisper-models\faster-whisper-small-ms")
DEFAULT_FFMPEG = Path(r"D:\_tool\ffmpeg\bin\ffmpeg.exe")
DEFAULT_ARTIFACTS = TRAINING_ROOT / "artifacts" / "video"


def parse_time(value: str | float | int) -> float:
    """``"01:23"`` / ``"1:02:03"`` / ``12.5`` → 秒。"""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{rest:05.2f}"
    return f"{minutes:02d}:{rest:05.2f}"


def ffmpeg() -> str:
    if DEFAULT_FFMPEG.is_file():
        return str(DEFAULT_FFMPEG)
    return "ffmpeg"


def decode_audio(video: Path, start: float, end: float) -> "object":
    """用 ffmpeg 把区间解成 16kHz 单声道 float32（不经 PyAV，避免版本不兼容）。"""
    import numpy as np

    command = [ffmpeg(), "-v", "quiet", "-ss", f"{start:.3f}"]
    if end > start:
        command += ["-to", f"{end:.3f}"]
    command += ["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败: {completed.stderr.decode('utf-8', 'replace')[:400]}")
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def extract_frames(video: Path, out_dir: Path, *, start: float, end: float,
                   every: float, max_edge: int = 960) -> list[dict]:
    """按固定间隔抽帧（``every`` 秒一张），返回时间戳清单。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    step = max(1.0, float(every))
    moment = start
    index = 0
    while moment <= end:
        target = out_dir / f"f-{index:04d}-{moment:07.1f}s.jpg"
        command = [
            ffmpeg(), "-v", "quiet", "-ss", f"{moment:.3f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "3",
            "-vf", f"scale='min({max_edge},iw)':-2",
            "-y", str(target),
        ]
        subprocess.run(command, capture_output=True)
        if target.is_file() and target.stat().st_size > 0:
            frames.append({"t": round(moment, 2), "path": str(target),
                           "bytes": target.stat().st_size})
        index += 1
        moment += step
    return frames


def transcribe(video: Path, *, start: float, end: float, model_dir: Path,
               language: str, beam_size: int) -> dict:
    from faster_whisper import WhisperModel

    audio = decode_audio(video, start, end)
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio, language=language, beam_size=beam_size, vad_filter=True,
        condition_on_previous_text=False,
    )
    rows = []
    for segment in segments:
        rows.append({
            "start": round(start + float(segment.start), 2),
            "end": round(start + float(segment.end), 2),
            "text": str(segment.text).strip(),
        })
    return {
        "schema": "gladiator.video-asr/v1",
        "video": str(video),
        "range": [start, end],
        "language": language,
        "duration": round(float(getattr(info, "duration", 0.0)) + start, 2),
        "segments": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频 → 语音转写 + 关键帧")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--start", default="0", help="起点（00:41 或秒）")
    parser.add_argument("--end", default="0", help="终点；0 表示到结尾")
    parser.add_argument("--out", default=None, help="产物目录（默认 artifacts/video/<stem>）")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="faster-whisper 模型目录")
    parser.add_argument("--language", default="zh", help="语音语言")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--frame-every", type=float, default=0.0,
                        help="每 N 秒抽一帧；0 表示不抽帧")
    parser.add_argument("--no-asr", action="store_true", help="只抽帧，不做语音转写")
    parser.add_argument("--seed-minutes", type=float, default=0.0,
                        help="距上次抽帧超过该分钟数就跳过（断点续跑用）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        print(f"视频不存在: {video}", file=sys.stderr)
        return 2
    start = parse_time(args.start)
    end = parse_time(args.end)
    out_dir = Path(args.out) if args.out else DEFAULT_ARTIFACTS / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    notes: dict = {
        "schema": "gladiator.video-notes/v1",
        "video": str(video),
        "range": [start, end],
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if not args.no_asr:
        model_dir = Path(args.model_dir)
        if not (model_dir / "model.bin").is_file():
            print(f"缺少模型权重: {model_dir / 'model.bin'}", file=sys.stderr)
            return 2
        asr = transcribe(video, start=start, end=end, model_dir=model_dir,
                         language=args.language, beam_size=args.beam_size)
        (out_dir / "asr.json").write_text(
            json.dumps(asr, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"[{fmt_time(row['start'])}] {row['text']}" for row in asr["segments"]]
        (out_dir / "asr.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        notes["asr"] = {"segments": len(asr["segments"]),
                        "chars": sum(len(row["text"]) for row in asr["segments"])}

    if args.frame_every > 0:
        frames = extract_frames(video, out_dir / "frames", start=start,
                                end=end or start, every=args.frame_every)
        (out_dir / "frames.json").write_text(
            json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
        notes["frames"] = len(frames)

    notes["elapsedSeconds"] = round(time.time() - started, 1)
    (out_dir / "notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_dir),
                      "elapsedSeconds": notes["elapsedSeconds"],
                      "asr": notes.get("asr"), "frames": notes.get("frames")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
