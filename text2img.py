import torch
from diffusers import StableDiffusionPipeline

# Định nghĩa tham số
rand_seed = torch.manual_seed(42) # Thường dùng trong các bài toán sinh ảnh
num_inference_steps = 25 # Số lần vector sinh từ text đi qua pipeline để sinh ra ảnh
guidance_scale = 0.75 # Độ follow theo text mà người dùng nhập vào
height = 512
width = 512


# Danh sách model
# https://huggingface.co/models

model_list = ["nota-ai/bk-sdm-small",
              "CompVis/stable-diffusion-v1-4",
              "runwayml/stable-diffusion-v1-5",
              "prompthero/openjourney",
              "hakurei/waifu-diffusion",
              "stabilityai/stable-diffusion-2-1",
              "dreamlike-art/dreamlike-photoreal-2.0"
              ]

def create_pipeline(model_name = model_list[0]):
    # Nếu máy có GPU CUDA
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            use_safetensors = True
        ).to("cuda")
    elif torch.backends.mps.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("mps")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
    return pipeline

def text2image(prompt, pipeline):
    images = pipeline(
        prompt,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        generator = rand_seed,
        num_images_per_request = 1,
        height = height,
        width = width
    ).images
    return images[0]


