# ref: https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video#cerspensezeroscopev2576w-cerspensezeroscopev2xl

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 这一段非常影响torch.onnx.export的输出结果
# pipe.enable_model_cpu_offload()


# 加了这段代码，生成效率也会降低很多
# memory optimization
# pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
# pipe.enable_vae_slicing()


import gc
import os
from collections import OrderedDict
from copy import copy
from typing import List, Optional, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import PIL
import tensorrt as trt
import torch
from huggingface_hub import snapshot_download
from onnx import shape_inference
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel, UNet3DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)

from diffusers.pipelines.text_to_video_synthesis import (
    VideoToVideoSDPipeline
)

from diffusers.schedulers import DDIMScheduler, KarrasDiffusionSchedulers
from diffusers.utils import DIFFUSERS_CACHE, logging


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def device_view(t):
    return cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=torch_to_numpy_dtype_dict[t.dtype])


def preprocess_image(image):
    """
    image: torch.Tensor
    """
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous()
    return 2.0 * image - 1.0



class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        workspace_size=0,
    ):
        logger.warning(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs["preview_features"] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs["preview_features"].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(fp16=fp16, profiles=[p], load_timing_cache=timing_cache, **config_kwargs),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.warning(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


class Optimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph



class BaseModel:
    def __init__(self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77):
        self.model = model
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        # self.min_image_shape = 64  # min image resolution: 256x256
        # self.max_image_shape = 256  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8
        self.min_frame_num = 6
        self.max_frame_num = 12

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        return self.model

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, frame_num, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, frame_num, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        
        min_frame_num = frame_num if static_shape else self.min_frame_num
        max_frame_num = frame_num if static_shape else self.max_frame_num
        
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
            min_frame_num, 
            max_frame_num
        )



def getOnnxPath(model_name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, model_name + (".opt" if opt else "") + ".onnx")


def getEnginePath(model_name, engine_dir):
    return os.path.join(engine_dir, model_name + ".plan")


def build_engines(
    models: dict,
    engine_dir,
    onnx_dir,
    onnx_opset,
    opt_frame_num, 
    opt_image_height,
    opt_image_width,
    opt_batch_size=1,
    force_engine_rebuild=False,
    static_batch=False,
    static_shape=True,
    enable_preview=False,
    enable_all_tactics=False,
    timing_cache=None,
    max_workspace_size=0,
):
    built_engines = {}
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)

    # Export models to ONNX
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        if force_engine_rebuild or not os.path.exists(engine_path):
            logger.warning("Building Engines...")
            logger.warning("Engine build can take a while to complete")
            onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = getOnnxPath(model_name, onnx_dir)
            if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                if force_engine_rebuild or not os.path.exists(onnx_path):
                    logger.warning(f"Exporting model: {onnx_path}")
                    if model_name == 'unet':
                        model = model_obj.get_model()
                        model.cpu()
                        model_obj.device = 'cpu'
                        with torch.inference_mode():
                            inputs = model_obj.get_sample_input(opt_batch_size, opt_frame_num, opt_image_height, opt_image_width)
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=18,
                                do_constant_folding=True,
                                input_names=model_obj.get_input_names(),
                                output_names=model_obj.get_output_names(),
                                # dynamic_axes=model_obj.get_dynamic_axes(),
                            )
                    else:
                        model = model_obj.get_model()
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = model_obj.get_sample_input(opt_batch_size, opt_frame_num, opt_image_height, opt_image_width)
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=onnx_opset,
                                do_constant_folding=True,
                                input_names=model_obj.get_input_names(),
                                output_names=model_obj.get_output_names(),
                                dynamic_axes=model_obj.get_dynamic_axes(),
                            )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.warning(f"Found cached model: {onnx_path}")

                # Optimize onnx
                if model_name != 'unet':
                    if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                        logger.warning(f"Generating optimizing model: {onnx_opt_path}")
                        onnx_opt_graph = model_obj.optimize(onnx.load(onnx_path))
                        onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        logger.warning(f"Found cached optimized model: {onnx_opt_path} ")

    # Build TensorRT engines
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
        onnx_opt_path = getOnnxPath(model_name, onnx_dir)

        if model_name == 'unet':
            # cmd = '/opt/tensorrt/bin/trtexec --onnx={0} --saveEngine={1} --fp16'.format(onnx_path, engine_path)
            # os.system(cmd)
            if force_engine_rebuild or not os.path.exists(engine.engine_path):
                engine.build(
                    onnx_path,
                    fp16=True,
                    input_profile=model_obj.get_input_profile(
                        opt_batch_size,
                        opt_frame_num, 
                        opt_image_height,
                        opt_image_width,
                        static_batch=static_batch,
                        static_shape=static_shape,
                    ),
                    enable_preview=enable_preview,
                    timing_cache=timing_cache,
                    workspace_size=max_workspace_size,
                )
            built_engines[model_name] = engine
        else:
            if force_engine_rebuild or not os.path.exists(engine.engine_path):
                engine.build(
                    onnx_opt_path,
                    fp16=True,
                    input_profile=model_obj.get_input_profile(
                        opt_batch_size,
                        opt_frame_num, 
                        opt_image_height,
                        opt_image_width,
                        static_batch=static_batch,
                        static_shape=static_shape,
                    ),
                    enable_preview=enable_preview,
                    timing_cache=timing_cache,
                    workspace_size=max_workspace_size,
                )
            built_engines[model_name] = engine

    # Load and activate TensorRT engines
    for model_name, model_obj in models.items():
        engine = built_engines[model_name]
        engine.load()
        engine.activate()

    return built_engines


def runEngine(engine, feed_dict, stream):
    return engine.infer(feed_dict, stream)



class CLIP(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(CLIP, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, frame_num, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, frame_num, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, frame_num, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size,  frame_num, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        return opt_onnx_graph


def make_CLIP(model, device, max_batch_size, embedding_dim, inpaint=False):
    return CLIP(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class UNet(BaseModel):
    def __init__(
        self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77, unet_dim=4
    ):
        super(UNet, self).__init__(
            model=model,
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self):
        return ["latent"]

    # def get_dynamic_axes(self):
    #     return {
    #         "sample": {0: "2B", 2: "H", 3: "W"},
    #         "encoder_hidden_states": {0: "2B"},
    #         "latent": {0: "2B", 2: "H", 3: "W"},
    #     }
    
    # def get_dynamic_axes(self):
    #     return {
    #         "sample": {0: "2B", 2:'F', 3: "H", 4: "W"},
    #         "encoder_hidden_states": {0: "2B"},
    #         "latent": {0: "2B", 2:'F', 3: "H", 4: "W"},
    #     }
    
    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 3: "H", 4: "W"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 3: "H", 4: "W"},
        }


    def get_input_profile(self, batch_size, frame_num, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
            min_frame_num, 
            max_frame_num
        ) = self.get_minmax_dims(batch_size, frame_num, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (2 * min_batch, self.unet_dim, min_frame_num, min_latent_height, min_latent_width),
                (2 * batch_size, self.unet_dim, frame_num, latent_height, latent_width),
                (2 * max_batch, self.unet_dim, max_frame_num, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size, frame_num, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, frame_num, latent_height, latent_width),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, frame_num, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, frame_num, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        dtype = torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, frame_num, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            # torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.tensor([1], dtype=torch.int64, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        )


def make_UNet(model, device, max_batch_size, embedding_dim, inpaint=False):
    return UNet(
        model,
        fp16=True,
        device=device,
        max_batch_size=max_batch_size,
        embedding_dim=embedding_dim,
        unet_dim=(9 if inpaint else 4),
    )


class VAE(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(VAE, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    def get_input_profile(self, batch_size, frame_num, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, frame_num, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch * frame_num, 4, min_latent_height, min_latent_width),
                (batch_size * frame_num, 4, latent_height, latent_width),
                (max_batch * frame_num, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, frame_num, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, frame_num, latent_height, latent_width),
            "images": (batch_size, 3, frame_num, image_height, image_width),
        }

    def get_sample_input(self, batch_size,  frame_num, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size * frame_num, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


def make_VAE(model, device, max_batch_size, embedding_dim, inpaint=False):
    return VAE(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vae_encoder = model

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoder(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(VAEEncoder, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "VAE encoder"

    def get_model(self):
        vae_encoder = TorchVAEEncoder(self.model)
        return vae_encoder

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {"images": {0: "B", 2: "8H", 3: "8W"}, "latent": {0: "B", 2: "H", 3: "W"}}

    def get_input_profile(self, batch_size, frame_num, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
            _,
            _
        ) = self.get_minmax_dims(batch_size, frame_num, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch * frame_num, 3, min_image_height, min_image_width),
                (batch_size * frame_num, 3, image_height, image_width),
                (max_batch * frame_num, 3, max_image_height, max_image_width),
            ]
        }

    def get_shape_dict(self, batch_size,  frame_num, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size * frame_num, 3, image_height, image_width),
            "latent": (batch_size * frame_num, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size,  frame_num, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size * frame_num, 3, image_height, image_width, dtype=torch.float32, device=self.device)


def make_VAEEncoder(model, device, max_batch_size, embedding_dim, inpaint=False):
    return VAEEncoder(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)




# vae = pipe.vae
# text_encoder = pipe.text_encoder
# tokenizer = pipe.tokenizer
# unet = pipe.unet

# stages = ["clip", "unet", "vae", "vae_encoder"]


class TensorRTVideoToVideoSDPipeline(VideoToVideoSDPipeline):
    def __init__(        
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        stages=["clip", "unet", "vae", "vae_encoder"],
        frame_num = 8, 
        image_height: int = 512,
        image_width: int = 512,
        max_batch_size: int = 1,
        # ONNX export parameters
        onnx_opset: int = 17,
        onnx_dir: str = "onnx",
        # TensorRT engine build parameters
        engine_dir: str = "engine",
        build_preview_features: bool = True,
        force_engine_rebuild: bool = False,
        timing_cache: str = "timing_cache",        
        ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler
        )

        self.vae.forward = self.vae.decode

        # stages = ['unet']
        # stages = ["vae", "vae_encoder"]
        self.stages = stages
        self.image_height, self.image_width = image_height, image_width
        self.frame_num = frame_num
        self.inpaint = False
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.timing_cache = timing_cache
        self.build_static_batch = False
        self.build_dynamic_shape = False
        self.build_preview_features = build_preview_features

        self.max_batch_size = max_batch_size
        # TODO: Restrict batch size to 4 for larger image dimensions as a WAR for TensorRT limitation.
        # if self.build_dynamic_shape or self.image_height > 512 or self.image_width > 512:
        #     self.max_batch_size = 4

        self.stream = None  # loaded in loadResources()
        self.models = {}  # loaded in __loadModels()
        self.engine = {}  # loaded in build_engines()
        
        
    def __loadModels(self):
        # Load pipeline models
        self.embedding_dim = self.text_encoder.config.hidden_size
        models_args = {
            "device": self.torch_device,
            "max_batch_size": self.max_batch_size,
            "embedding_dim": self.embedding_dim,
            "inpaint": self.inpaint,
        }
        if "clip" in self.stages:
            self.models["clip"] = make_CLIP(self.text_encoder, **models_args)
        if "unet" in self.stages:
            self.models["unet"] = make_UNet(self.unet, **models_args)
        if "vae" in self.stages:
            self.models["vae"] = make_VAE(self.vae, **models_args)
        if "vae_encoder" in self.stages:
            self.models["vae_encoder"] = make_VAEEncoder(self.vae, **models_args)    
        
    @classmethod
    def set_cached_folder(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)

        cls.cached_folder = (
            pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path)
            else snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        )
        

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings: bool = False):
        super().to(torch_device, silence_dtype_warnings=silence_dtype_warnings)

        self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)
        self.timing_cache = os.path.join(self.cached_folder, self.timing_cache)

        # set device
        self.torch_device = self._execution_device
        logger.warning(f"Running inference on device: {self.torch_device}")

        # load models
        self.__loadModels()

        # build engines
        self.engine = build_engines(
            self.models,
            self.engine_dir,
            self.onnx_dir,
            self.onnx_opset,
            opt_frame_num=self.frame_num, 
            opt_image_height=self.image_height,
            opt_image_width=self.image_width,
            force_engine_rebuild=self.force_engine_rebuild,
            static_batch=self.build_static_batch,
            static_shape=not self.build_dynamic_shape,
            enable_preview=self.build_preview_features,
            timing_cache=self.timing_cache,
        )

        return self



pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", 
                                         torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# v2v_pipe = TensorRTVideoToVideoSDPipeline(
#     pipe.vae, 
#     pipe.text_encoder,
#     pipe.tokenizer, 
#     pipe.unet, 
#     pipe.scheduler,
#     frame_num=20, 
#     image_height=576, 
#     image_width=1024
# )

v2v_pipe = TensorRTVideoToVideoSDPipeline(
    pipe.vae, 
    pipe.text_encoder,
    pipe.tokenizer, 
    pipe.unet, 
    pipe.scheduler,
    frame_num=12, 
    image_height=512, 
    image_width=512
)

v2v_pipe.set_cached_folder('/workspace/code/aigc/text2video/AIGCVideo/Demo/zeroscope/xxx')
v2v_pipe.to('cuda')


print('congratulations!')