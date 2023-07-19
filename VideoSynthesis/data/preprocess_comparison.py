import cv2
import torch
import numpy as np
import nvcv
import cvcuda
from PIL import Image
from cupyx.profiler import benchmark


def CenterCrop_Comparison(img_path, batch_size):
    def CenterCrop(rgb, size=256):
        w, h = rgb[0].size
        assert min(w, h) >= size
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        rgb = [u.crop((x1, y1, x1 + size, y1 + size)) for u in rgb]
        return rgb

    def CenterCrop_cvcuda(rgb, size=256):
        rgb = cvcuda.center_crop(rgb, (size, size))
        return rgb

    for bs in batch_size:
        # original
        image = Image.open(img_path)
        batch_image = [image] * bs
        original_res = benchmark(CenterCrop, (batch_image, 256, ), n_warmup=50, n_repeat=50)

        # cvcuda
        img = cv2.imread(img_path)
        batch_image = np.asarray([img] * bs)
        torch_tensor = torch.from_numpy(batch_image).cuda(0)
        cvcuda_input_tensor = cvcuda.as_tensor(torch_tensor, "NHWC")
        cvcuda_res = benchmark(CenterCrop_cvcuda, (cvcuda_input_tensor, 256, ), n_warmup=50, n_repeat=50)

        print(f"batch size: {bs} \n original result:{original_res} \n cvcuda res: {cvcuda_res} \n")



def CenterCropV2_Comparison(img_path, batch_size):
    def CenterCropV2(img, size):
        # fast resize
        while min(img[0].size) >= 2 * size:
            img = [u.resize((u.width // 2, u.height // 2), resample=Image.BOX) for u in img]
        scale = size / min(img[0].size)
        img = [u.resize((round(scale * u.width), round(scale * u.height)), resample=Image.BICUBIC) for u in img]
        
        # center crop
        x1 = (img[0].width - size) // 2
        y1 = (img[0].height - size) // 2
        img = [u.crop((x1, y1, x1 + size, y1 + size)) for u in img]
        return img

    def CenterCropV2_cvcuda(img, img_size, size):
        # fast resize
        while min(img_size[0]) >= 2 * size:
            img_size = img_size // 2
            cvcuda.resize(img, [tuple(l) for l in img_size])

        scale = size / (min(img_size[0]))
        img_size = np.round(scale * img_size).astype(np.int32)
        img = cvcuda.resize(img, [tuple(l) for l in img_size], cvcuda.Interp.CUBIC)

        # center crop
        img = [cvcuda.center_crop(cvcuda.as_tensor(im), (size, size)) for im in img]
        return img


    for bs in batch_size:
        # original
        image = Image.open(img_path)
        batch_image = [image] * bs
        original_res = benchmark(CenterCropV2, (batch_image, 256, ), n_warmup=50, n_repeat=50)


        # cvcuda
        img = cv2.imread(img_path)
        img_size = np.asarray([img.shape[:2]] * bs)
        torch_tensor = torch.from_numpy(img).cuda(0)

        batch_image = nvcv.ImageBatchVarShape(bs)
        for _ in range(bs): 
            batch_image.pushback(nvcv.as_image(torch_tensor))

        cvcuda_res = benchmark(CenterCropV2_cvcuda, (batch_image, img_size, 256), n_warmup=50, n_repeat=50)
        print(f"batch size: {bs} \n original result:{original_res} \n cvcuda res: {cvcuda_res} \n")


def Resize_Comparison(img_path, batch_size):
    def Resize(rgb, size=256):
        rgb = [u.resize((size, size), Image.BILINEAR) for u in rgb]
        return rgb

    def Resize_cvcuda(rgb, size):
        rgb = cvcuda.resize(rgb, size, cvcuda.Interp.LINEAR)
        return rgb
    
    for bs in batch_size:
        # original
        image = Image.open(img_path)
        batch_image = [image] * bs
        original_res = benchmark(Resize, (batch_image, 256, ), n_warmup=50, n_repeat=50)


        # cvcuda
        img = cv2.imread(img_path)
        size = [(256, 256) for _b in range(bs)]
        torch_tensor = torch.from_numpy(img).cuda(0)

        batch_image = nvcv.ImageBatchVarShape(bs)
        for _ in range(bs): 
            batch_image.pushback(nvcv.as_image(torch_tensor))

        cvcuda_res = benchmark(Resize_cvcuda, (batch_image, size), n_warmup=50, n_repeat=50)
        print(f"batch size: {bs} \n original result:{original_res} \n cvcuda res: {cvcuda_res} \n")


def Normalize_Comparison(img_path, batch_size):
    def Normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        rgb = rgb.clone()
        rgb.clamp_(0, 1)
        if not isinstance(mean, torch.Tensor):
            mean = rgb.new_tensor(mean).view(-1)
        if not isinstance(std, torch.Tensor):
            std = rgb.new_tensor(std).view(-1)
        rgb.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
        return rgb
    
    def Normalize_cvcuda(rgb, mean, std):
        rgb = cvcuda.normalize(
            rgb,
            base=mean,
            scale=std,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV
        )
        return rgb



    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv2.imread(image_path)

    for bs in batch_size:
        batch_image = (np.asarray([img] * bs) / 255.0).astype(np.float32)

        # original
        original_input = torch.Tensor(batch_image).permute(0, 3, 1, 2)
        original_res = benchmark(Normalize, (original_input, mean, std), n_warmup=50, n_repeat=50)

        # cvcuda
        cvcuda_input_tensor = cvcuda.as_tensor(
            torch.from_numpy(batch_image).cuda(0), 
            "NHWC"
        )
        base_tensor = cvcuda.as_tensor(
            torch.reshape(torch.Tensor(mean), (1,1,1,3)).cuda(),
            "NHWC"
        )
        stddev_tensor = cvcuda.as_tensor(
            torch.reshape(torch.Tensor(std), (1,1,1,3)).cuda(),
            "NHWC"
        )
        cvcuda_res = benchmark(Normalize_cvcuda, (cvcuda_input_tensor, base_tensor, stddev_tensor), n_warmup=50, n_repeat=50)
        print(f"batch size: {bs} \n original result:{original_res} \n cvcuda res: {cvcuda_res} \n")




batch_size = [1, 10, 30, 50]
image_path = 'sample.jpg'

Normalize_Comparison(image_path, batch_size)