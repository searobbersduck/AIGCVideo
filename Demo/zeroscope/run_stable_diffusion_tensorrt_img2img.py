# ref: https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#tensorrt-image2image-stable-diffusion-pipeline

import requests
from io import BytesIO
from PIL import Image
import torch
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionImg2ImgPipeline

# Use the DDIMScheduler scheduler here instead
scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
                                            subfolder="scheduler")


pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                custom_pipeline="/workspace/code/aigc/deps/diffusers/examples/community/stable_diffusion_tensorrt_img2img.py",
                                                revision='fp16',
                                                torch_dtype=torch.float16,
                                                scheduler=scheduler,)

# re-use cached folder to save ONNX models and TensorRT Engines
pipe.set_cached_folder("stabilityai/stable-diffusion-2-1", revision='fp16',)

pipe = pipe.to("cuda")

url = "https://pajoca.com/wp-content/uploads/2022/09/tekito-yamakawa-1.png"
response = requests.get(url)
input_image = Image.open(BytesIO(response.content)).convert("RGB")

prompt = "photorealistic new zealand hills"
image = pipe(prompt, image=input_image, strength=0.75,).images[0]
image.save('tensorrt_img2img_new_zealand_hills.png')