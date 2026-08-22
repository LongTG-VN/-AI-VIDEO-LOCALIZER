import json
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


class MediaService:
    def __init__(self, ffprobe_bin: str = "ffprobe", ffmpeg_bin: str = "ffmpeg"):
        self.ffprobe_bin = ffprobe_bin
        self.ffmpeg_bin = ffmpeg_bin

    def probe(self, path: str | Path) -> dict:
        target = Path(path)
        if not target.exists():
            raise MediaError(f"Media file does not exist: {target}")
        cmd = [self.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(target)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise MediaError("ffprobe is not installed or not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise MediaError(exc.stderr.strip() or "ffprobe failed") from exc
        data = json.loads(result.stdout)
        video_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
        duration_raw = data.get("format", {}).get("duration") or video_stream.get("duration")
        return {"duration": float(duration_raw) if duration_raw is not None else None, "width": video_stream.get("width"), "height": video_stream.get("height"), "raw": data}

    def extract_audio(self, source: str | Path, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.ffmpeg_bin, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise MediaError("ffmpeg is not installed or not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise MediaError(exc.stderr.strip() or "ffmpeg audio extraction failed") from exc
        return destination
