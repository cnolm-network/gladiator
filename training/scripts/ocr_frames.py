"""用 Windows 内置 OCR（离线、无 GPU）把截图里的中文批量转成文字。

为什么不用视觉模型：本机没有 CUDA，4B VL 跑不动；Windows 自带 OCR 对
**幻灯片/字幕/计分板**这类高对比文字足够，而且零下载。图片里的文字与数字
仍以人眼（或决策模型看图）为准，OCR 只是省 context 的索引。

用法::

    python scripts/ocr_frames.py training/artifacts/video/0/frames --out frames.ocr.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _ocr_engine(language: str):
    from winsdk.windows.media.ocr import OcrEngine

    engine = OcrEngine.try_create_from_language(
        __import__("winsdk.windows.globalization", fromlist=["Language"]).Language(language))
    if engine is None:
        engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        raise RuntimeError("系统没有可用的 OCR 语言包（设置 → 时间和语言 → 语言 → 中文 → 可选功能 → OCR）")
    return engine


async def _read_text(engine, path: Path) -> str:
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.storage import FileAccessMode, StorageFile

    file = await StorageFile.get_file_from_path_async(str(path.resolve()))
    stream = await file.open_async(FileAccessMode.READ)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    result = await engine.recognize_async(bitmap)
    return "\n".join(line.text for line in result.lines)


def ocr_images(paths: list[Path], language: str = "zh-Hans-CN") -> dict[str, str]:
    engine = _ocr_engine(language)
    loop = asyncio.new_event_loop()
    try:
        return {str(path): loop.run_until_complete(_read_text(engine, path)) for path in paths}
    finally:
        loop.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows OCR 批量识别")
    parser.add_argument("target", help="图片文件或目录")
    parser.add_argument("--out", default=None, help="输出文本（默认打印到 stdout）")
    parser.add_argument("--language", default="zh-Hans-CN")
    args = parser.parse_args(argv)
    target = Path(args.target)
    paths = sorted(target.glob("*.jpg")) if target.is_dir() else [target]
    results = ocr_images(paths, language=args.language)
    text = "\n\n".join(f"### {Path(key).name}\n{value}" for key, value in results.items())
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "images": len(paths), "out": args.out}, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
