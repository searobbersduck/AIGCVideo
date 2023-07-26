# [zeroscope](https://huggingface.co/cerspense)

## 环境配置

**docker和conda在整个测试过程中都会用到，都需要进行配置**

**docker 环境的配置如下：**
```
docker run --shm-size=10gb --gpus all -it --name TENSORRT_TEXT2VIDEO -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /home/rtx/workspace/docker_workspace:/workspace nvcr.io/nvidia/tensorrt:23.05-py3 bash
```

注意`pytorch`需要安装`nightly`版本

其它，根据需要进行安装：[install the libraries required](https://huggingface.co/cerspense/zeroscope_v2_XL#usage-in-%F0%9F%A7%A8-diffusers)

不要安装`xformers`，如果要安装，请进行编译安装，不要破坏`pytorch`的`nightly`环境。

<br><br>

**conda 环境的配置如下：**

为什么需要`conda`环境？
* TRT的运行需要`pytorch nightly`的环境，`xformers`的`pip`直接安装会破坏`pytorch`环境，为了方便，这里会单独创建`conda`环境；

参照: [install the libraries required](https://huggingface.co/cerspense/zeroscope_v2_XL#usage-in-%F0%9F%A7%A8-diffusers)

```
conda create --name=zeroscope python=3.10

conda activate zeroscope

pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate torch
pip install xformers
```

<br><br>
***
<br><br>

## 相关链接

* [cerspense/zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL)
* [cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w)
* [Text-to-video synthesis](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video#diffusers.VideoToVideoSDPipeline.__call__)
    * 这里也可以找到上面两个链接的代码调用


<br>

## 代码实现逻辑

首先对比[`stable_diffusion_tensorrt_img2img.py`](https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_tensorrt_img2img.py)和[`pipeline_stable_diffusion_img2img.py`](https://github.com/huggingface/diffusers/blob/716286f19ddd9eb417113e064b538706884c8e73/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L92)的实现。

在`diffuser`下的相对路径如下：
`stable_diffusion_tensorrt_img2img.py`: `examples/community/stable_diffusion_tensorrt_img2img.py`
`pipeline_stable_diffusion_img2img.py`: `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py` 

`stable_diffusion_tensorrt_img2img.py` 参照：[TensorRT Image2Image Stable Diffusion Pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#tensorrt-image2image-stable-diffusion-pipeline)

`video2video`: [`pipeline_text_to_video_synth_img2img.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py), 
在`diffuser`下的相对路径如下：
`pipeline_text_to_video_synth_img2img.py`: `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py`

<br>

## 代码运行

<br>

切换到`Demo/zeroscope`目录

<br>

**测试TRT实现的性能：**

* 测试`video generate end2end pipeline`的运行时间，**这里的是基于TRT 16bit的pipeline的运行时间**， 运行：`CUDA_VISIBLE_DEVICES=1 python run_stable_diffusion_tensorrt_v2v.py`

* 运行过程如下：

    ```
    Running inference on device: cuda:0
    Loading TensorRT engine: /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/clip.plan
    [I] Loading bytes from /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/clip.plan
    [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
    Loading TensorRT engine: /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/unet.plan
    [I] Loading bytes from /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/unet.plan
    Loading TensorRT engine: /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/vae.plan
    [I] Loading bytes from /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/vae.plan
    Loading TensorRT engine: /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/vae_encoder.plan
    [I] Loading bytes from /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx/engine/vae_encoder.plan
    /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/run_stable_diffusion_tensorrt_v2v.py:231: DeprecationWarning: Use get_tensor_dtype instead.
    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
    /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/run_stable_diffusion_tensorrt_v2v.py:232: DeprecationWarning: Use get_tensor_mode instead.
    if self.engine.binding_is_input(binding):
    /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/run_stable_diffusion_tensorrt_v2v.py:233: DeprecationWarning: Use set_input_shape instead.
    self.context.set_binding_shape(idx, shape)
    /workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/run_stable_diffusion_tensorrt_v2v.py:230: DeprecationWarning: Use get_tensor_shape instead.
    shape = self.engine.get_binding_shape(binding)
    /usr/local/lib/python3.10/dist-packages/diffusers/configuration_utils.py:134: FutureWarning: Accessing config attribute `steps_offset` directly via 'DPMSolverMultistepScheduler' object attribute is deprecated. Please access 'steps_offset' over 'DPMSolverMultistepScheduler's config object instead, e.g. 'scheduler.config.steps_offset'.
    deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 49/49 [00:23<00:00,  2.05it/s]
    time elapsed:   25.0778s
    congratulations!

    ```
* TRT运行时间`25.0778s`

<br>

**测试pytorch fp16的运行时间**

* 切换到`conda`环境
* 运行`CUDA_VISIBLE_DEVICES=1 python run_zeroscope_v2_xl.py`
* 运行过程如下：
    ```
    CUDA_VISIBLE_DEVICES=1 python run_zeroscope_v2_xl.py
    unet/diffusion_pytorch_model.safetensors not found
    Loading pipeline components...: 100%|████████████████████████████████████████████████████████████████████████████| 5/5 [00:24<00:00,  4.99s/it]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:15<00:00,  3.25it/s]
    unet/diffusion_pytorch_model.safetensors not found
    Loading pipeline components...: 100%|████████████████████████████████████████████████████████████████████████████| 5/5 [00:27<00:00,  5.46s/it]
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:29<00:00,  1.70it/s]
    time elapsed:   32.3041s

    ```
* pytorch fp16运行时间`32.3041s`

<br>

加速比：`28.8%`

