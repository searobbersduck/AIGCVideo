# [zeroscope](https://huggingface.co/cerspense)

## 相关链接

* [cerspense/zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL)
* [cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w)

## 

首先对比[`stable_diffusion_tensorrt_img2img.py`](https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_tensorrt_img2img.py)和[`pipeline_stable_diffusion_img2img.py`](https://github.com/huggingface/diffusers/blob/716286f19ddd9eb417113e064b538706884c8e73/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L92)的实现。

在`diffuser`下的相对路径如下：
`stable_diffusion_tensorrt_img2img.py`: `examples/community/stable_diffusion_tensorrt_img2img.py`
`pipeline_stable_diffusion_img2img.py`: `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py` 

`stable_diffusion_tensorrt_img2img.py` 参照：[TensorRT Image2Image Stable Diffusion Pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#tensorrt-image2image-stable-diffusion-pipeline)

`video2video`: [`pipeline_text_to_video_synth_img2img.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py), 
在`diffuser`下的相对路径如下：
`pipeline_text_to_video_synth_img2img.py`: `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py`

