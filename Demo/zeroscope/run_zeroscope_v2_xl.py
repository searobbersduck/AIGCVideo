import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

from PIL import Image

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPTextModel, CLIPTokenizer

def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images


def preprocess_video(video):
    supported_formats = (np.ndarray, torch.Tensor, PIL.Image.Image)

    if isinstance(video, supported_formats):
        video = [video]
    elif not (isinstance(video, list) and all(isinstance(i, supported_formats) for i in video)):
        raise ValueError(
            f"Input is in incorrect format: {[type(i) for i in video]}. Currently, we only support {', '.join(supported_formats)}"
        )

    if isinstance(video[0], PIL.Image.Image):
        video = [np.array(frame) for frame in video]

    if isinstance(video[0], np.ndarray):
        video = np.concatenate(video, axis=0) if video[0].ndim == 5 else np.stack(video, axis=0)

        if video.dtype == np.uint8:
            video = np.array(video).astype(np.float32) / 255.0

        if video.ndim == 4:
            video = video[None, ...]

        video = torch.from_numpy(video.transpose(0, 4, 1, 2, 3))

    elif isinstance(video[0], torch.Tensor):
        video = torch.cat(video, axis=0) if video[0].ndim == 5 else torch.stack(video, axis=0)

        # don't need any preprocess if the video is latents
        channel = video.shape[1]
        if channel == 4:
            return video

        # move channels before num_frames
        video = video.permute(0, 2, 1, 3, 4)

    # normalize video
    video = 2.0 * video - 1.0

    return video


pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=12).frames
video_path = export_to_video(video_frames)
video_path


pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]



num_images_per_prompt = 1

batch_size = 1

device = pipe._execution_device

guidance_scale = 15.0
do_classifier_free_guidance = guidance_scale > 1.0

negative_prompt = None
prompt_embeds = None
negative_prompt_embeds = None

cross_attention_kwargs = None

text_encoder_lora_scale = (
    cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
)


# prompt_embeds.shape: torch.Size([2, 77, 1024])
prompt_embeds = pipe._encode_prompt(
    prompt, 
    device, 
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt, 
    prompt_embeds=prompt_embeds, 
    negative_prompt_embeds=negative_prompt_embeds, 
    lora_scale=text_encoder_lora_scale
)

# video.shape: torch.Size([1, 3, 12, 576, 1024])
video = preprocess_video(video)

num_inference_steps = 50
strength = 0.6

pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, strength, device)
latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

generator = None

# 5. Prepare latent variables
# latents.shape: torch.Size([1, 3, 12, 576, 1024])
latents = pipe.prepare_latents(video, latent_timestep, batch_size, prompt_embeds.dtype, device, generator)


eta = 0.0
extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order

video_frames = pipe(prompt, video=video, strength=0.6).frames
video_path = export_to_video(video_frames)
video_path