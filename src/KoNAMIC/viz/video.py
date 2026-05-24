import subprocess


def frames_to_video_ffmpeg(frames_dir, output_path, fps=30):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "step_%04d_overlay.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)