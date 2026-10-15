import subprocess
import wave

import numpy as np


def finish_video(raw, out, audio, smooth, hd):
    from imageio_ffmpeg import get_ffmpeg_exe
    if audio is None and not (smooth or hd):
        raw.replace(out)
        return
    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw)]
    wav = None
    if audio is not None:
        samples, rate = audio
        wav = raw.with_suffix(".wav")
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        cmd += ["-i", str(wav)]
    filters = []
    if smooth:
        filters.append("minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    if hd:
        filters.append("scale=1024:576:flags=lanczos")
    if filters:
        cmd += ["-vf", ",".join(filters), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    if wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    subprocess.run(cmd + ["-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    if wav:
        wav.unlink()
