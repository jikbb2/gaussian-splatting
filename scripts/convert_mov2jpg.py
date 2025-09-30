#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert .MOV to .JPG frames while preserving (mapping) metadata into EXIF
- Requires: ffmpeg, exiftool
- What it does:
  1) Extract frames with ffmpeg
  2) Read MOV metadata via exiftool (JSON) and process it
  3) Map processed metadata (Make, Model, FocalLength, GPS, Sensor Size etc.) to each JPG
  4) (Optional) Calculate and write per-frame DateTimeOriginal
"""

import argparse
import json
import re
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import uuid


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")
    return proc.stdout

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_step_seconds(fps: float, every_n_frames: int = None, every_seconds: float = None) -> float:
    if every_seconds is not None:
        return float(every_seconds)
    if every_n_frames is not None:
        return float(every_n_frames) / float(fps if fps > 0 else 30.0)
    return 1.0 / float(fps if fps > 0 else 30.0)

def ffprobe_fps(video: Path) -> float:
    def parse_fraction(frac: str):
        if not frac or "/" not in frac:
            try: return float(frac)
            except (ValueError, TypeError): return None
        n, d = frac.split("/")
        try:
            n, d = float(n), float(d)
            return n / d if d != 0 else 0
        except ValueError:
            return None

    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video)
    ]).strip()
    fps = parse_fraction(out)
    if fps and fps > 0:
        return fps

    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video)
    ]).strip()
    fps = parse_fraction(out)
    return fps if fps and fps > 0 else 30.0

def exiftool_json(video: Path) -> dict:
    out = run(["exiftool", "-json", "-n", str(video)])
    data = json.loads(out)
    return data[0] if data else {}

def parse_exiftool_datetime(dt_str: str):
    if not dt_str: return None, None, None
    iso_like = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", dt_str).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_like)
        offset_str = dt.strftime('%z')
        if offset_str:
            offset_str = f"{offset_str[:-2]}:{offset_str[-2:]}"
        return dt.strftime("%Y:%m:%d %H:%M:%S"), dt.tzinfo, offset_str
    except Exception:
        m = re.match(r"^(\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2})", dt_str)
        return (m.group(1), None, None) if m else (None, None, None)

def extract_frames(video: Path, out_dir: Path, quality=2,
                   every_n_frames: int = None, every_seconds: float = None, prefix: str = ""):
    ensure_dir(out_dir)
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vsync", "0", "-q:v", str(quality), "-start_number", "0"]
    if every_seconds is not None:
        cmd += ["-vf", f"fps=1/{every_seconds}"]
    elif every_n_frames is not None and every_n_frames > 1:
        cmd += ["-vf", f"select='not(mod(n,{every_n_frames}))'"]

    image_format = f"{prefix}_%04d.jpg"
    cmd += [str(out_dir / image_format)]
    run(cmd)

def process_metadata(meta: dict) -> dict:
    """Exiftool JSON 데이터를 입력받아 필요한 메타데이터를 추출하고 계산합니다."""
    make = meta.get("Make")
    model = meta.get("Model")
    focal_length_35mm = meta.get("CameraFocalLength35mmEquivalent") or meta.get("FocalLengthIn35mmFormat")
    image_width_px = meta.get("ImageWidth")

    focal_length = None
    lens_model_str = meta.get("CameraLensModel") or meta.get("LensModel")
    if lens_model_str:
        match = re.search(r"(\d+(\.\d+)?)mm", lens_model_str)
        if match:
            focal_length = float(match.group(1))

    sensor_width = None
    focal_plane_x_res = None
    if focal_length and focal_length_35mm and focal_length > 0:
        try:
            crop_factor = focal_length_35mm / focal_length
            sensor_width = 36.0 / crop_factor
            # FocalPlaneXResolution 계산 (이미지 너비 픽셀 수 / 센서 너비 mm)
            if image_width_px and sensor_width > 0:
                focal_plane_x_res = image_width_px / sensor_width
        except ZeroDivisionError:
            pass

    serial_number = meta.get("BodySerialNumber") or str(uuid.uuid5(uuid.NAMESPACE_DNS, str(meta.get("SourceFile"))))
    
    gps = {
        "GPSLatitude": meta.get("GPSLatitude"),
        "GPSLongitude": meta.get("GPSLongitude"),
        "GPSAltitude": meta.get("GPSAltitude"),
    }

    return {
        "Make": make,
        "Model": model,
        "FocalLength": focal_length,
        "FocalLengthIn35mmFilm": focal_length_35mm,
        "SensorWidth": 10.629541, 
        "FocalPlaneXResolution": focal_plane_x_res, # EXIF에 기록할 값
        "LensSerialNumber": serial_number,
        "Software": meta.get("Software"),
        **gps
    }

def build_exif_assignments(meta: dict, dto_string: str, offset_str: str) -> list:
    """가공된 메타데이터로 exiftool 할당 명령어를 생성합니다."""
    assigns = ["-n"]

    if meta.get("Make"): assigns.append(f"-EXIF:Make={meta['Make']}")
    if meta.get("Model"): assigns.append(f"-EXIF:Model={meta['Model']}")
    if meta.get("Software"): assigns.append(f"-EXIF:Software={meta['Software']}")
    if meta.get("LensSerialNumber"): 
        assigns.append(f"-EXIF:LensSerialNumber={meta['LensSerialNumber']}") 
    else:
        print("not have serial number")
    if meta.get("FocalLength") is not None: assigns.append(f"-EXIF:FocalLength={meta['FocalLength']}")
    if meta.get("FocalLengthIn35mmFilm") is not None:
        try:
            efl35_int = int(round(float(meta['FocalLengthIn35mmFilm'])))
            assigns.append(f"-EXIF:FocalLengthIn35mmFormat={efl35_int}")
        except (ValueError, TypeError):
            pass
    if meta.get("FocalPlaneXResolution") is not None:
        assigns.append(f"-EXIF:FocalPlaneXResolution={meta['FocalPlaneXResolution']}")
        # ResolutionUnit을 'mm'로 설정 (3=cm, 4=mm, 5=um)
        assigns.append("-EXIF:FocalPlaneResolutionUnit=4")

    assigns.append("-EXIF:Orientation=1")

    if meta.get("GPSLatitude") is not None: assigns.append(f"-EXIF:GPSLatitude={meta['GPSLatitude']}")
    if meta.get("GPSLongitude") is not None: assigns.append(f"-EXIF:GPSLongitude={meta['GPSLongitude']}")
    if meta.get("GPSAltitude") is not None: assigns.append(f"-EXIF:GPSAltitude={meta['GPSAltitude']}")

    if dto_string:
        assigns.append(f"-EXIF:DateTimeOriginal={dto_string}")
        assigns.append(f"-EXIF:CreateDate={dto_string}")
        if offset_str:
            assigns.append(f"-EXIF:OffsetTimeOriginal={offset_str}")

    return assigns

def write_exif_to_jpg(jpg_path: Path, assignments: list):
    cmd = ["exiftool", "-overwrite_original", "-P"] + assignments + [str(jpg_path)]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="MOV -> JPG frames with EXIF mapped")
    ap.add_argument("inputs", nargs="+", help="Input .MOV file(s) (glob patterns ok)")
    ap.add_argument("--out", required=True, help="Output directory for frames")
    ap.add_argument('--prefix', type=str, default='', help='A prefix to add to each frame filename (e.g., "aug_").')
    ap.add_argument("--quality", type=int, default=2, help="JPEG quality (2=high, 31=low)")
    ap.add_argument("--datetime", action="store_true", help="Write per-frame DateTimeOriginal")
    ap.add_argument("--sample-every-frames", type=int, help="Pick every Nth frame")
    ap.add_argument("--sample-every-seconds", type=float, help="Pick one frame every S seconds")
    args = ap.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    for inpat in args.inputs:
        patterns = list(Path().glob(inpat)) if any(c in inpat for c in "*?[]") else [Path(inpat)]
        for mov in sorted(patterns):
            if not mov.exists():
                print(f"[WARN] Skip (not found): {mov}")
                continue

            out_dir = out_root
            print(f"==> Processing: {mov} -> {out_dir}")
            ensure_dir(out_dir)

            raw_meta = exiftool_json(mov)
            meta = process_metadata(raw_meta)

            creation = (raw_meta.get("CreationDate") or raw_meta.get("MediaCreateDate") or raw_meta.get("CreateDate"))
            dto_base_str, _, offset_str = parse_exiftool_datetime(creation) if creation else (None, None, None)

            fps = ffprobe_fps(mov)
            print(f"   FPS: {fps:.3f}")

            extract_frames(mov, out_dir, quality=args.quality,
                           every_n_frames=args.sample_every_frames,
                           every_seconds=args.sample_every_seconds, prefix=args.prefix)

            jpgs = sorted(out_dir.glob(f"{args.prefix}_*.jpg"))
            if not jpgs:
                print("[WARN] No frames extracted; check ffmpeg output.")
                continue

            step_seconds = compute_step_seconds(fps, args.sample_every_frames, args.sample_every_seconds)
            for idx, jpg in enumerate(jpgs):
                dto_string = dto_base_str
                if args.datetime and dto_base_str:
                    base_dt = datetime.strptime(dto_base_str, "%Y:%m:%d %H:%M:%S")
                    dt_offset = timedelta(seconds=(idx * step_seconds))
                    dto_string = (base_dt + dt_offset).strftime("%Y:%m:%d %H:%M:%S")
                
                assignments = build_exif_assignments(meta, dto_string, offset_str)
                write_exif_to_jpg(jpg, assignments)

            print(f"   Wrote {len(jpgs)} frames with EXIF.")

if __name__ == "__main__":
    main()
