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
  9. loop_video: NVENC GPU encoder with automatic CPU fallback if NVENC unavailable.
  10. handler: torch.cuda.empty_cache() in finally to prevent VRAM accumulation.

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


def _nvenc_usable():
    """Check if NVIDIA hardware encoder is available on this machine."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q"], stderr=subprocess.DEVNULL
        ).decode()
        return "Encoder" in out
    except Exception:
        return False


# --- GLOBAL VARIABLES FOR OCCLUSION AND FACE TRACKING ---
CURRENT_FRAME_COUNTER = 0
CURRENT_OCCLUSION_MASK = []
LAST_VALID_BBOX = None
LAST_VALID_LMK = None

try:
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    from DeepCache import DeepCacheSDHelper
    _LATENTSYNC_AVAILABLE = True

    # --- MONKEY PATCHES TO ELIMINATE LATENTSYNC CPU BOTTLENECKS ---
    import latentsync.utils.util
    import latentsync.pipelines.lipsync_pipeline
    import latentsync.utils.image_processor
    import latentsync.utils.face_detector

    # Patch 1: Bypass CPU-bound FPS resampling in read_video (we prep video at 25 FPS on GPU)
    _orig_read_video = latentsync.utils.util.read_video
    def optimized_read_video(video_path, change_fps=True, use_decord=True):
        print(f"[latentsync patch] read_video called for {video_path}, forcing change_fps=False to avoid CPU re-encoding", flush=True)
        return _orig_read_video(video_path, change_fps=False, use_decord=use_decord)
    latentsync.utils.util.read_video = optimized_read_video
    latentsync.pipelines.lipsync_pipeline.read_video = optimized_read_video
    latentsync.utils.image_processor.read_video = optimized_read_video

    # Patch 2: Use GPU (NVENC) for the final video merge in lipsync_pipeline with CPU fallback
    _orig_subprocess_run = subprocess.run
    def patched_subprocess_run(args, **kwargs):
        if isinstance(args, str) and "ffmpeg" in args and "-c:v libx264" in args:
            if _nvenc_usable():
                print("[latentsync patch] Replacing libx264 with h264_nvenc for final video encoding", flush=True)
                nvenc_args = args.replace("-c:v libx264 -crf 18", "-c:v h264_nvenc -rc vbr -cq 20 -preset fast")
                res = _orig_subprocess_run(nvenc_args, **kwargs)
                if res.returncode != 0:
                    print(f"[latentsync patch] NVENC final merge failed (code {res.returncode}). Retrying with CPU libx264 fallback...", flush=True)
                    return _orig_subprocess_run(args, **kwargs)
                return res
            else:
                print("[latentsync patch] NVENC unavailable, keeping libx264 for final video encoding", flush=True)
        return _orig_subprocess_run(args, **kwargs)
    latentsync.pipelines.lipsync_pipeline.subprocess.run = patched_subprocess_run

    # Patch 3: Dynamic occlusion masking based on hand detection
    _orig_prepare_masks = latentsync.utils.image_processor.ImageProcessor.prepare_masks_and_masked_images
    def patched_prepare_masks(self, images, affine_transform=False):
        global CURRENT_FRAME_COUNTER, CURRENT_OCCLUSION_MASK
        pixel_values, masked_pixel_values, masks = _orig_prepare_masks(self, images, affine_transform=affine_transform)

        batch_size = len(images)
        for k in range(batch_size):
            global_idx = CURRENT_FRAME_COUNTER + k
            if global_idx < len(CURRENT_OCCLUSION_MASK) and CURRENT_OCCLUSION_MASK[global_idx]:
                print(f"[occlusion patch] Frame {global_idx} is occluded by a hand. Bypassing lipsync.", flush=True)
                masks[k] = 1.0  # 1.0 mask means keep original pixels in paste_surrounding_pixels_back
                masked_pixel_values[k] = pixel_values[k]

        CURRENT_FRAME_COUNTER += batch_size
        return pixel_values, masked_pixel_values, masks
    latentsync.utils.image_processor.ImageProcessor.prepare_masks_and_masked_images = patched_prepare_masks

    # Patch 4: Safety check on affine_transform_video to prevent empty frames from causing stack trace crash
    _orig_affine_transform_video = latentsync.pipelines.lipsync_pipeline.LipsyncPipeline.affine_transform_video
    def patched_affine_transform_video(self, video_frames):
        try:
            return _orig_affine_transform_video(self, video_frames)
        except RuntimeError as e:
            if "stack expects a non-empty TensorList" in str(e):
                raise RuntimeError("No faces detected in the video. Please make sure the video contains a visible face.") from e
            raise e
    latentsync.pipelines.lipsync_pipeline.LipsyncPipeline.affine_transform_video = patched_affine_transform_video

    # Patch 5: Caching and auto-recovery for face landmarks to prevent 'Face not detected' crashes
    _orig_face_detector_call = latentsync.utils.face_detector.FaceDetector.__call__
    def patched_face_detector_call(self, frame, threshold=0.5):
        global LAST_VALID_BBOX, LAST_VALID_LMK
        bbox, lmk = _orig_face_detector_call(self, frame, threshold=threshold)

        if bbox is not None:
            LAST_VALID_BBOX = bbox
            LAST_VALID_LMK = lmk
            return bbox, lmk

        if LAST_VALID_BBOX is not None:
            # Re-use the cached landmarks to recover from temporary face detection loss
            return LAST_VALID_BBOX, LAST_VALID_LMK

        return None, None
    latentsync.utils.face_detector.FaceDetector.__call__ = patched_face_detector_call

    # Patch 6: Batched inference for LipsyncPipeline.__call__ to massive speedup
    def batched_call(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        from typing import Optional, Union, List, Callable
        from einops import rearrange
        import tqdm
        import soundfile as sf
        import shutil
        from latentsync.utils.image_processor import ImageProcessor
        from latentsync.utils.util import read_audio, write_video

        # We can pass custom batch_size via kwargs or read from env or default to 4
        inference_batch_size = kwargs.get("batch_size") or int(os.environ.get("INFERENCE_BATCH_SIZE", 4))
        print(f"[latentsync patch] Running batched inference with batch_size={inference_batch_size}", flush=True)

        is_train = self.unet.training
        self.unet.eval()

        # 0. Define call parameters
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        video_frames, original_video_frames, boxes, affine_matrices = self.affine_transform_video(video_path)
        audio_samples = read_audio(audio_path)

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            num_inferences = min(len(video_frames), len(whisper_chunks)) // num_frames
        else:
            num_inferences = len(video_frames) // num_frames

        synced_video_frames = []
        masked_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            1,
            num_frames * num_inferences,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        for i in tqdm.tqdm(range(0, num_inferences, inference_batch_size), desc="Doing inference..."):
            bs = min(inference_batch_size, num_inferences - i)
            
            # Prepare audio embeds for all bs chunks
            if self.unet.add_audio_layer:
                audio_embeds_list = []
                for k in range(bs):
                    idx = (i + k) * num_frames
                    chunk = torch.stack(whisper_chunks[idx : idx + num_frames])
                    audio_embeds_list.append(chunk)
                audio_embeds = torch.cat(audio_embeds_list, dim=0)
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    empty_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([empty_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            # Slice video frames for the bs chunks
            inference_video_frames = video_frames[i * num_frames : (i + bs) * num_frames]
            
            # Slice and reshape latents: shape (bs, c, f, h, w)
            latents = all_latents[:, :, i * num_frames : (i + bs) * num_frames]
            latents = rearrange(latents, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)

            # VAE encoding and mask preparation for the bs chunks (returns shape (bs * f, ...))
            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_video_frames, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # Reshape mask_latents and masked_image_latents to (bs or 2 * bs, c, f, h, w)
            if do_classifier_free_guidance:
                mask_uncond, mask_cond = mask_latents.chunk(2, dim=0)
                mask_uncond = rearrange(mask_uncond, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                mask_cond = rearrange(mask_cond, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                mask_latents = torch.cat([mask_uncond, mask_cond], dim=0)

                mi_uncond, mi_cond = masked_image_latents.chunk(2, dim=0)
                mi_uncond = rearrange(mi_uncond, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                mi_cond = rearrange(mi_cond, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                masked_image_latents = torch.cat([mi_uncond, mi_cond], dim=0)
            else:
                mask_latents = rearrange(mask_latents, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                masked_image_latents = rearrange(masked_image_latents, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)

            # 8. Prepare image latents
            image_latents = self.prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            if do_classifier_free_guidance:
                il_uncond, il_cond = image_latents.chunk(2, dim=0)
                il_uncond = rearrange(il_uncond, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                il_cond = rearrange(il_cond, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)
                image_latents = torch.cat([il_uncond, il_cond], dim=0)
            else:
                image_latents = rearrange(image_latents, "1 c (b f) h w -> b c f h w", b=bs, f=num_frames)

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            masked_video_frames.append(masked_pixel_values)

        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        )
        masked_video_frames = self.restore_video(
            torch.cat(masked_video_frames), original_video_frames, boxes, affine_matrices
        )

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)

    latentsync.pipelines.lipsync_pipeline.LipsyncPipeline.__call__ = batched_call

except ImportError as e:
    print(f"Warning: Could not import LatentSync dependencies: {e}. Running in stub/compatibility mode.", flush=True)
    _LATENTSYNC_AVAILABLE = False


R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "https://acabe0325acfdba5f87564c12f31ea9a.r2.cloudflarestorage.com")
R2_BUCKET = os.environ.get("R2_BUCKET", "lawyerdigest")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE", "https://cdn.sttiz.com")

# Pipeline global variables
PIPE = None
CONFIG = None
DTYPE = None
DEEPCACHE_HELPER = None
DEVICE = "cuda"

CONFIG_PATH = os.path.join(LATENTSYNC_DIR, "configs", "unet", "stage2_512.yaml")
SCHEDULER_DIR = os.path.join(LATENTSYNC_DIR, "configs")
UNET_CKPT = os.path.join(LATENTSYNC_DIR, "checkpoints", "latentsync_unet.pt")
WHISPER_TINY = os.path.join(LATENTSYNC_DIR, "checkpoints", "whisper", "tiny.pt")
WHISPER_SMALL = os.path.join(LATENTSYNC_DIR, "checkpoints", "whisper", "small.pt")
MASK_PATH = os.path.join(LATENTSYNC_DIR, "latentsync", "utils", "mask.png")


def load_pipe():
    global PIPE, CONFIG, DTYPE, DEEPCACHE_HELPER
    if not _LATENTSYNC_AVAILABLE:
        print("[latentsync] load_pipe stub: LatentSync dependencies not loaded", flush=True)
        return None
    if PIPE is not None:
        return PIPE

    print("[latentsync] Loading LatentSync pipeline...", flush=True)
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    CONFIG = OmegaConf.load(CONFIG_PATH)

    # FP16 precision choice
    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    DTYPE = torch.float16 if is_fp16 else torch.float32
    print(f"[latentsync] Selected precision: {DTYPE}", flush=True)

    scheduler = DDIMScheduler.from_pretrained(SCHEDULER_DIR)

    if CONFIG.model.cross_attention_dim == 768:
        whisper_path = WHISPER_SMALL
    else:
        whisper_path = WHISPER_TINY

    if not os.path.exists(whisper_path):
        raise FileNotFoundError(f"Whisper model not found: {whisper_path}")

    audio_encoder = Audio2Feature(
        model_path=whisper_path,
        device=DEVICE,
        num_frames=CONFIG.data.num_frames,
        audio_feat_length=CONFIG.data.audio_feat_length,
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # UNet
    if not os.path.exists(UNET_CKPT):
        raise FileNotFoundError(f"UNet checkpoint not found: {UNET_CKPT}")

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(CONFIG.model),
        UNET_CKPT,
        device="cpu",
    )
    unet = unet.to(dtype=DTYPE)

    # Fix mask path
    CONFIG.data.mask_image_path = MASK_PATH

    # Build Pipeline
    PIPE = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(DEVICE)

    # Pre-init DeepCache helper (enable/disable happens per request)
    DEEPCACHE_HELPER = DeepCacheSDHelper(pipe=PIPE)

    print("[latentsync] Pipeline loaded successfully!", flush=True)
    return PIPE



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




def _video_codec_args():
    """Return FFmpeg codec args: h264_nvenc if GPU available, libx264 fallback."""
    if _nvenc_usable():
        print("[ffmpeg] Using h264_nvenc (GPU)", flush=True)
        return ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "20", "-preset", "fast"]
    else:
        print("[ffmpeg] NVENC unavailable — falling back to libx264 (CPU)", flush=True)
        return ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"]


def _run_ffmpeg_command(cmd_args, out_path):
    print(f"[prep] Running FFmpeg command: {' '.join(cmd_args)}", flush=True)
    res = subprocess.run(cmd_args, capture_output=True, text=True)

    # Verify success: return code 0 and file exists with size > 1000 bytes
    success = (res.returncode == 0) and os.path.exists(out_path) and (os.path.getsize(out_path) > 1000)

    if not success:
        # Check if NVENC was used and try CPU fallback
        has_nvenc = any("h264_nvenc" in arg for arg in cmd_args)
        if has_nvenc:
            print("[prep] GPU NVENC command failed or produced empty file. Retrying with CPU (libx264) fallback...", flush=True)
            print(f"[prep] GPU fail code: {res.returncode}\nstdout: {res.stdout}\nstderr: {res.stderr}", flush=True)

            # Map args to replace NVENC with libx264 parameters
            new_args = []
            skip = 0
            for i, arg in enumerate(cmd_args):
                if skip > 0:
                    skip -= 1
                    continue
                if arg == "-c:v" and i + 1 < len(cmd_args) and cmd_args[i+1] == "h264_nvenc":
                    new_args += ["-c:v", "libx264"]
                    skip = 1
                elif arg in ("-rc", "-cq"):
                    # skip these flags and their arguments
                    skip = 1
                elif arg == "-preset" and i + 1 < len(cmd_args) and cmd_args[i+1] == "fast":
                    new_args += ["-preset", "veryfast"]
                    skip = 1
                else:
                    new_args.append(arg)

            # Remove partial output if it exists
            if os.path.exists(out_path):
                try: os.remove(out_path)
                except Exception: pass

            print(f"[prep] Running CPU fallback command: {' '.join(new_args)}", flush=True)
            res2 = subprocess.run(new_args, capture_output=True, text=True)
            fallback_success = (res2.returncode == 0) and os.path.exists(out_path) and (os.path.getsize(out_path) > 1000)
            if not fallback_success:
                print(f"[prep] CPU fallback failed too. code: {res2.returncode}\nstdout: {res2.stdout}\nstderr: {res2.stderr}", flush=True)
                raise RuntimeError(f"FFmpeg command failed on both GPU (NVENC) and CPU. Stderr: {res2.stderr[-500:]}")
        else:
            print(f"[prep] FFmpeg command failed. code: {res.returncode}\nstdout: {res.stdout}\nstderr: {res.stderr}", flush=True)
            raise RuntimeError(f"FFmpeg command failed: {res.stderr[-500:]}")


def prepare_input_video(video_path, target_duration):
    """Ensure video is exactly target_duration, 25 FPS, 720p width scaling, and encoded with NVENC/CPU fallback."""
    video_dur = get_duration(video_path)
    codec_args = _video_codec_args()
    out = f"{CACHE}/prepared_{uuid.uuid4().hex[:8]}.mp4"

    # Scale filter: scale to max width 720 (Instagram format) maintaining aspect ratio, force 25 FPS
    scale_filter = ["-vf", "scale='min(720,iw)':-2"]

    # If duration is close enough, we still run a quick GPU pass to ensure 25 FPS and correct codec
    if abs(video_dur - target_duration) < 0.5:
        print(f"[prep] Video duration matches target. Normalizing to 720p, 25 FPS and GPU codec...", flush=True)
        _run_ffmpeg_command([
            "ffmpeg", "-y", "-i", video_path, "-t", f"{target_duration:.3f}",
            "-r", "25", *scale_filter, *codec_args,
            "-c:a", "aac", "-b:a", "128k",
            out
        ], out)
        return out

    # If it needs trimming
    if video_dur > target_duration:
        print(f"[prep] Video is longer than audio. Trimming and normalizing to 720p, 25 FPS...", flush=True)
        _run_ffmpeg_command([
            "ffmpeg", "-y", "-i", video_path, "-t", f"{target_duration:.3f}",
            "-r", "25", *scale_filter, *codec_args,
            "-c:a", "aac", "-b:a", "128k",
            out
        ], out)
        return out

    # Looping logic
    print(f"[prep] Video is shorter than audio. Looping and normalizing to 720p, 25 FPS...", flush=True)
    loops = int(target_duration / max(video_dur, 1)) + 2
    concat_file = f"{CACHE}/concat_{uuid.uuid4().hex[:8]}.txt"
    with open(concat_file, "w") as f:
        for _ in range(loops):
            f.write(f"file '{video_path}'\n")

    try:
        _run_ffmpeg_command([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-t", f"{target_duration:.3f}",
            "-r", "25", *scale_filter, *codec_args,
            "-c:a", "aac", "-b:a", "128k",
            out
        ], out)
    finally:
        try: os.remove(concat_file)
        except Exception: pass
    return out


def detect_video_occlusion(video_path):
    """Detect hand-mouth occlusion using MediaPipe Hands and InsightFace landmarks."""
    import cv2
    import numpy as np

    occlusions = []
    if not _LATENTSYNC_AVAILABLE:
        return occlusions

    try:
        import mediapipe as mp
        from latentsync.utils.face_detector import FaceDetector

        # Temp FaceDetector instance on GPU (same config as the main pipeline)
        print("[occlusion] Initializing FaceDetector on CUDA for hand occlusion check...", flush=True)
        face_detector = FaceDetector(device="cuda")

        print("[occlusion] Initializing MediaPipe Hands...", flush=True)
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[occlusion] Warning: Failed to open video {video_path}", flush=True)
            return occlusions

        print("[occlusion] Analyzing frames...", flush=True)
        t_start = time.time()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face & 106-point landmarks
            bbox, lmk = face_detector(frame_rgb)
            if lmk is None:
                # No face detected in this frame, treat as no occlusion
                occlusions.append(False)
                frame_idx += 1
                continue

            # Mouth landmarks: 52 to 71 in 106-point system
            mouth_lmks = lmk[52:72]
            min_x, min_y = np.min(mouth_lmks, axis=0)
            max_x, max_y = np.max(mouth_lmks, axis=0)

            # 15% safety margin
            mw = max_x - min_x
            mh = max_y - min_y
            min_x -= int(mw * 0.15)
            max_x += int(mw * 0.15)
            min_y -= int(mh * 0.15)
            max_y += int(mh * 0.15)

            # Run MediaPipe Hands
            results = hands_detector.process(frame_rgb)
            is_occluded = False

            if results.multi_hand_landmarks:
                h_h, h_w, _ = frame.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        px = int(landmark.x * h_w)
                        py = int(landmark.y * h_h)
                        # Check if landmark is inside the mouth bounding box
                        if min_x <= px <= max_x and min_y <= py <= max_y:
                            is_occluded = True
                            break
                    if is_occluded:
                        break

            occlusions.append(is_occluded)
            frame_idx += 1

        cap.release()
        hands_detector.close()
        print(f"[occlusion] Finished in {time.time() - t_start:.2f}s. Detected {sum(occlusions)} occluded frames out of {len(occlusions)}.", flush=True)

    except Exception as e:
        print(f"[occlusion] Error during detection: {e}. Skipping occlusion bypass.", flush=True)
        traceback.print_exc()

    return occlusions


def run_latentsync(video_path, audio_path, inference_steps, guidance_scale, seed, deepcache_interval, batch_size=4):
    global CURRENT_FRAME_COUNTER, CURRENT_OCCLUSION_MASK, LAST_VALID_BBOX, LAST_VALID_LMK
    CURRENT_FRAME_COUNTER = 0
    LAST_VALID_BBOX = None
    LAST_VALID_LMK = None
    CURRENT_OCCLUSION_MASK = detect_video_occlusion(video_path)

    output = f"{CACHE}/output_{uuid.uuid4().hex[:8]}.mp4"

    if not _LATENTSYNC_AVAILABLE:
        print("[latentsync] running in stub mode (dependencies not available). Copying input video.", flush=True)
        import shutil
        shutil.copy(video_path, output)
        return output

    temp_dir = f"{CACHE}/temp_{uuid.uuid4().hex[:8]}"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[latentsync] starting native pipeline inference: steps={inference_steps}, guidance={guidance_scale}, seed={seed}, deepcache_interval={deepcache_interval}, batch_size={batch_size}", flush=True)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup DeepCache dynamically
    if DEEPCACHE_HELPER is not None:
        if deepcache_interval > 1:
            print(f"[latentsync] DeepCache enabled with interval={deepcache_interval}", flush=True)
            DEEPCACHE_HELPER.set_params(cache_interval=deepcache_interval, cache_branch_id=0)
            DEEPCACHE_HELPER.enable()
        else:
            print("[latentsync] DeepCache disabled for this run", flush=True)
            if hasattr(DEEPCACHE_HELPER, "function_dict"):
                try:
                    DEEPCACHE_HELPER.disable()
                except Exception as e:
                    print(f"[latentsync] Warning: Failed to disable DeepCache: {e}", flush=True)

    # Heartbeat thread — print every 30s so RunPod doesn't mark worker idle
    stop = threading.Event()
    def heartbeat():
        t = 0
        while not stop.is_set():
            stop.wait(30)
            t += 30
            if stop.is_set(): break
            print(f"[latentsync] pipeline inference heartbeat t={t}s", flush=True)
    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()

    try:
        pipe = load_pipe()
        pipe(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output,
            num_frames=CONFIG.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=DTYPE,
            width=CONFIG.data.resolution,
            height=CONFIG.data.resolution,
            mask_image_path=CONFIG.data.mask_image_path,
            temp_dir=temp_dir,
            batch_size=batch_size,
        )
    finally:
        stop.set()
        th.join(timeout=1)
        # Cleanup pipeline temp_dir
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    if not os.path.exists(output) or os.path.getsize(output) < 1000:
        raise RuntimeError("LatentSync produced no/empty output.")

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

        # default_steps, default_deepcache, and default_batch_size from env or hard defaults
        default_steps = int(os.environ.get("DEFAULT_INFERENCE_STEPS", 20))
        default_cache = int(os.environ.get("DEFAULT_DEEPCACHE_INTERVAL", 3))
        default_batch_size = int(os.environ.get("DEFAULT_INFERENCE_BATCH_SIZE", 4))

        inference_steps = max(10, min(50, int(job_input.get("inference_steps", default_steps))))
        deepcache_interval = max(1, min(10, int(job_input.get("deepcache_interval", default_cache))))
        batch_size = max(1, min(16, int(job_input.get("batch_size", default_batch_size))))
        guidance_scale = max(1.0, min(3.0, float(job_input.get("guidance_scale", 1.5))))
        seed = int(job_input.get("seed") or random.randint(1, 10**6))

        print(f"[{job_id}] mode={return_mode} steps={inference_steps} guidance={guidance_scale} seed={seed} deepcache_interval={deepcache_interval} batch_size={batch_size}", flush=True)

        print(f"[{job_id}] Downloading inputs...", flush=True)
        video_path = download_or_decode(video_src, "mp4")
        audio_path = download_or_decode(audio_src, "mp3")

        audio_dur = get_duration(audio_path)
        video_dur = get_duration(video_path)
        print(f"[{job_id}] audio={audio_dur:.1f}s video={video_dur:.1f}s", flush=True)

        if audio_dur < 1:
            return {"error": "Audio file unreadable or shorter than 1s"}
        if video_dur < 0.1:
            return {"error": "Downloaded video file is unreadable, empty or shorter than 0.1s"}

        # Always run prepare_input_video to guarantee 25 FPS, duration alignment, and GPU-compatible codec
        print(f"[{job_id}] Preparing and normalizing input video...", flush=True)
        video_path = prepare_input_video(video_path, audio_dur)

        prepared_dur = get_duration(video_path)
        print(f"[{job_id}] Video prepared: {prepared_dur:.1f}s at 25 FPS", flush=True)
        if prepared_dur < 0.1:
            raise RuntimeError(f"Prepared video file is unreadable, empty or invalid: {video_path}")

        print(f"[{job_id}] Running LatentSync...", flush=True)
        t1 = time.time()
        synced = run_latentsync(video_path, audio_path, inference_steps, guidance_scale, seed, deepcache_interval, batch_size)
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

    finally:
        # Free VRAM after every request (success or failure)
        if _LATENTSYNC_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# Warm up LatentSync pipeline on worker startup
try:
    load_pipe()
except Exception as e:
    print(f"CRITICAL: Failed to preload LatentSync pipeline: {e}", flush=True)
    traceback.print_exc()

runpod.serverless.start({"handler": handler})
