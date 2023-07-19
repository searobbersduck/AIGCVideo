import torch
import time
import tensorrt as trt
import cupy as cp
import numpy as np
import open_clip
from cupyx.profiler import benchmark

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-H-14", 
    device=torch.device('cpu'), 
    pretrained="videocomposer-main/model_weights/open_clip_pytorch_model.bin"
)


def measure_token_embedding():
    dummy_input = torch.randn(list(model.token_embedding.parameters())[0].shape)
    torch.onnx.export(
        model, 
        dummy_input, 
        "clip_model.onnx",
        export_params=True,
        input_names = ["input"]
    )   


def measure_trans_model():
    dummy_input = torch.randn([1] + list(list(model.parameters())[0].shape))
    torch.onnx.export(
        model.transformer, 
        dummy_input, 
        "clip_model.onnx",
        export_params=True,
        input_names = ["input"],
        output_names = ["output"],
        dynamic_axes={'input': {0 : 'batch_size'},
                        'output': {0: 'batch_size'}}
    )    


def evaluate_trans_model(batch_size):
    
    n_warmup = 10
    n_repeat = 10

    input_data = torch.randn([batch_size] + list(list(model.parameters())[0].shape))


    model.to(0)
    torch_input = input_data.to(0)

    def pytorch_model():
        model.transformer(torch_input)

    
    input_data = torch.randn([batch_size] + list(list(model.parameters())[0].shape))

    with open("clip_model.trt", "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_data_trt = input_data.detach().cpu().numpy().ravel()

    d_input = cp.asarray(input_data_trt, dtype=cp.float32)
    output_shape = tuple(dim for dim in engine.get_binding_shape(1) if dim != -1)
    d_output = cp.empty(np.prod(output_shape), dtype=cp.float32)

    # Benchmarking
    pytorch_benchmark = benchmark(pytorch_model, (), n_warmup=n_warmup, n_repeat=n_repeat)

    trt_benchmark = benchmark(context.execute_v2, 
        ([int(d_input.data.ptr), int(d_output.data.ptr)], ), 
        n_warmup=n_warmup, n_repeat=n_repeat
    )
    
    
    print(pytorch_benchmark)
    print(trt_benchmark)


measure_trans_model()
# evaluate_trans_model(1)