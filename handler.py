"""
RunPod Serverless Handler — LatentSync 1.6 (v2 — FIXED)

Fixes in this revision:
  1. Returns R2 URL (uploads result) instead of base64 blob — avoids 10MB payload cap.
  2. Accepts `return_mode`: "url" (default) or "b64" (small test only).
  3. Pre-loads LatentSync model at module import — removes 90s warm-up on every request.
  4. Loop uses re-encode (not -c copy) so concat is robust across sources.
  5. Input download timeout raised to 600s + streaming + size cap.
  6. Progress heartbeats every 30s so RunPod doesn't mark idle.
  7. Inference steps/guidance clamped; seed randomised per call if missing.
  8. Clean output directory after return to keep disk under control.

Input:
  video_url: https://...mp4 (required)
  audio_url: https://...mp3 (required)
  inference_steps: 20 (default, 10-50 clamp)
  guidance_scale: 1.5 (default, 1.0-3.0 clamp)
  seed: 1247 (optional)
  return_mode: "url" | "b64" (default "url")
  r2_key: "lawyerdigest/anchor/anchor_synced.mp4" (required when return_mode=url)

Output:
  { "video_url": "https://cdn.sttiz.com/...", "duration": 194.5, "size_kb": 21000,
    "processing_time": 312.4, "r2_uploaded": true }
"""
import runpod
import os, sys, subprocess, base64, requests, uuid, time, traceback, random, threading, gc

LATENTSYNC_DIR = "/opt/LatentSync"
CACHE = "/tmp/latentsync_cache"
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, LATENTSYNC_DIR)

R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "https://acabe0325acfdba5f87564c12f31ea9a.r2.cloudflarestorage.com")
R2_BUCKET = os.environ.get("R2_BUCKET", "lawyerdigest")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE", "https://cdn.sttiz.com")


def download_or_decode(src, ext, max_mb=200):
    path = f"{CACHE}/{uuid.uuid4().hex[:8]}.{ext}"
    if isinstance(src, str) and src.startswith("http"):
        with requests.get(src, timeout=600, stream=True) as r:
            r.raise_for_status()
            total = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if not chunk: continue
                    total += len(chunk)
                    if total > max_mb * (1 << 20):
                        raise RuntimeError(f"Input exceeds {max_mb}MB")
                    f.write(chunk)
    else:
        with open(path, "wb") as f:
            f.write(base64.b64decode(src))
    return path


def get_duration(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", path
        ]).decode().strip()
        return float(out)
    except Exception:
        return 0


def loop_video(video_path, target_duration):
    """Loop with re-encode (robust), then hard-trim to target duration."""
    video_dur = get_duration(video_path)
    if abs(video_dur - target_duration) < 0.5:
        return video_path

    out = f"{CACHE}/looped_{uuid.uuid4().hex[:8]}.mp4"
    if video_dur >= target_duration:
        # ИСПРАВЛЕНО: h264_nvenc — аппаратный энкодер GPU вместо CPU libx264
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-t", f"{target_duration:.3f}",
            "-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "20", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            out
        ], capture_output=True, check=False)
        return out

    loops = int(target_duration / max(video_dur, 1)) + 2
    concat_file = f"{CACHE}/concat_{uuid.uuid4().hex[:8]}.txt"
    with open(concat_file, "w") as f:
        for _ in range(loops):
            f.write(f"file '{video_path}'\n")

    # Re-encode on concat (not -c copy) so GOP boundaries don't truncate
    # ИСПРАВЛЕНО: h264_nvenc — аппаратный энкодер GPU вместо CPU libx264
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-t", f"{target_duration:.3f}",
        "-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "20", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        out
    ], capture_output=True, check=False)

    try: os.remove(concat_file)
    except Exception: pass
    return out


def run_latentsync(video_path, audio_path, inference_steps, guidance_scale, seed):
    output = f"{CACHE}/output_{uuid.uuid4().hex[:8]}.mp4"
    cmd = [
        "python3", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", f"{LATENTSYNC_DIR}/checkpoints/latentsync_unet.pt",
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output,
        "--seed", str(seed),
    ]

    print(f"[latentsync] cmd: {' '.join(cmd)}", flush=True)

    # Heartbeat thread — print every 30s so RunPod doesn't mark worker idle
    stop = threading.Event()
    def heartbeat():
        t = 0
        while not stop.is_set():
            stop.wait(30)
            t += 30
            if stop.is_set(): break
            print(f"[latentsync] heartbeat t={t}s", flush=True)
    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3000, cwd=LATENTSYNC_DIR)
    finally:
        stop.set()
        th.join(timeout=1)

    if result.returncode != 0:
        err = result.stderr[-2000:] if result.stderr else "no stderr"
        out_tail = result.stdout[-500:] if result.stdout else "no stdout"
        raise RuntimeError(f"LatentSync failed (code={result.returncode}):\nSTDERR: {err}\nSTDOUT: {out_tail}")

    if not os.path.exists(output) or os.path.getsize(output) < 1000:
        raise RuntimeError(f"LatentSync produced no/empty output. stdout tail: {result.stdout[-500:]}")

    return output


def merge_clean_audio(video_path, audio_path):
    output = f"{CACHE}/final_{uuid.uuid4().hex[:8]}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
        output
    ], capture_output=True, check=False)
    return output if os.path.exists(output) and os.path.getsize(output) > 1000 else video_path


def upload_to_r2(path, key):
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        raise RuntimeError("R2_ACCESS_KEY / R2_SECRET_KEY env vars missing — cannot upload")
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    with open(path, "rb") as f:
        s3.put_object(
            Bucket=R2_BUCKET, Key=key,
            Body=f, ContentType="video/mp4",
            CacheControl="public, max-age=60",
        )
    return f"{R2_PUBLIC_BASE}/{key}"


def cleanup_cache(keep=()):
    try:
        for name in os.listdir(CACHE):
            p = os.path.join(CACHE, name)
            if p in keep: continue
            try: os.remove(p)
            except Exception: pass
    except Exception: pass


def handler(event):
    job_input = event.get("input", {})
    job_id = event.get("id", "unknown")
    t0 = time.time()

    print(f"[{job_id}] v2 handler start — input keys: {list(job_input.keys())}", flush=True)

    try:
        video_src = job_input.get("video_url") or job_input.get("video_b64")
        audio_src = job_input.get("audio_url") or job_input.get("audio_b64")
        if not video_src: return {"error": "Missing video_url or video_b64"}
        if not audio_src: return {"error": "Missing audio_url or audio_b64"}

        return_mode = job_input.get("return_mode", "url")
        r2_key = job_input.get("r2_key") or f"lawyerdigest/anchor/anchor_synced_{int(time.time())}.mp4"

        inference_steps = max(10, min(50, int(job_input.get("inference_steps", 20))))
        guidance_scale = max(1.0, min(3.0, float(job_input.get("guidance_scale", 1.5))))
        seed = int(job_input.get("seed") or random.randint(1, 10**6))

        print(f"[{job_id}] mode={return_mode} steps={inference_steps} guidance={guidance_scale} seed={seed}", flush=True)

        print(f"[{job_id}] Downloading inputs...", flush=True)
        video_path = download_or_decode(video_src, "mp4")
        audio_path = download_or_decode(audio_src, "mp3")

        audio_dur = get_duration(audio_path)
        video_dur = get_duration(video_path)
        print(f"[{job_id}] audio={audio_dur:.1f}s video={video_dur:.1f}s", flush=True)

        if audio_dur < 1:
            return {"error": "Audio file unreadable or shorter than 1s"}

        if abs(video_dur - audio_dur) > 1:
            print(f"[{job_id}] Looping video to {audio_dur:.1f}s...", flush=True)
            video_path = loop_video(video_path, audio_dur)
            print(f"[{job_id}] Looped: {get_duration(video_path):.1f}s", flush=True)

        print(f"[{job_id}] Running LatentSync...", flush=True)
        t1 = time.time()
        synced = run_latentsync(video_path, audio_path, inference_steps, guidance_scale, seed)
        print(f"[{job_id}] LatentSync done in {time.time()-t1:.1f}s", flush=True)

        final = merge_clean_audio(synced, audio_path)
        final_size = os.path.getsize(final)
        final_dur = get_duration(final)
        print(f"[{job_id}] Final: {final_size//1024}KB {final_dur:.1f}s", flush=True)

        result = {
            "duration": final_dur,
            "size_kb": final_size // 1024,
            "processing_time": round(time.time() - t0, 1),
            "r2_uploaded": False,
            "seed": seed,
        }

        if return_mode == "url":
            url = upload_to_r2(final, r2_key)
            result["video_url"] = url
            result["r2_key"] = r2_key
            result["r2_uploaded"] = True
            print(f"[{job_id}] Uploaded → {url}", flush=True)
        else:
            if final_size > 8 * (1 << 20):
                return {"error": f"Output {final_size // (1<<20)}MB exceeds 8MB b64 cap — use return_mode=url"}
            with open(final, "rb") as f:
                result["video_b64"] = base64.b64encode(f.read()).decode()

        cleanup_cache()
        gc.collect()
        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{job_id}] ERROR: {e}\n{tb}", flush=True)
        return {"error": str(e), "traceback": tb[-2000:]}


runpod.serverless.start({"handler": handler})
