import os
import argparse
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import sys

sys.path.append(os.getcwd())
from src.DLoRAL_model import Generator_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import PIL.Image
import math
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import glob
import torch
import gc
import cv2
from ram.models.ram_lora import ram
from ram import inference_ram as inference

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

center_crop = transforms.CenterCrop(128)
center_crop_gt = transforms.CenterCrop(512)


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"

    return validation_prompt


def extract_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)

    frame_number = 0
    success, frame = video_capture.read()
    frame_images = []

    while success:
        frame_dir = '{}'.format(video_path.split('.mp4')[0])
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        frame_filename = "frame_{:04d}.png".format(frame_number)
        cv2.imwrite('{}/{}'.format(frame_dir, frame_filename), frame)
        frame_images.append(os.path.join(frame_dir, frame_filename))
        success, frame = video_capture.read()
        frame_number += 1

    video_capture.release()
    return frame_images


def process_video_directory(input_directory):
    video_files = glob.glob(os.path.join(input_directory, "*.mp4"))
    all_video_data = []

    for video_file in video_files:
        frame_images = extract_frames(video_file)
        video_name = os.path.basename(video_file).split('.')[0]
        all_video_data.append((video_name, frame_images))

    return all_video_data

def compute_frame_difference_mask(frames):
    ambi_matrix = frames.var(dim=0)
    threshold = ambi_matrix.mean().item()
    mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
    frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
    return frame_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default=None, help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='the directory to save the output')
    parser.add_argument('--pretrained_path', type=str, default=None, help='path to a model state dict to be used')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='preset_models/stable-diffusion-2-1-base')
    parser.add_argument("--pretrained_model_path", type=str, default='preset_models/stable-diffusion-2-1-base')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--merge_and_unload_lora", default=False)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--load_cfr", action="store_true", )

    args = parser.parse_args()

    # initialize the model
    model = Generator_eval(args)
    model.set_eval()

    if os.path.isdir(args.input_image):
        all_video_data = process_video_directory(args.input_image)
    else:
        all_video_data = [(os.path.basename(args.input_image).split('.')[0], extract_frames(args.input_image))]

    # get ram model
    DAPE = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit='swin_l')
    DAPE.eval()
    DAPE.to("cuda")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # set weight type
    DAPE = DAPE.to(dtype=weight_dtype)
    model.vae = model.vae.to(dtype=weight_dtype)
    model.unet = model.unet.to(dtype=weight_dtype)
    model.cfr_main_net = model.cfr_main_net.to(dtype=weight_dtype)

    if args.stages == 0:
        model.unet.set_adapter(['default_encoder_consistency', 'default_decoder_consistency', 'default_others_consistency'])
    else:
        model.unet.set_adapter(['default_encoder_quality', 'default_decoder_quality',
                                'default_others_quality',
                                'default_encoder_consistency', 'default_decoder_consistency',
                                'default_others_consistency'])
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)
    frame_num = 2
    frame_overlap = 1

    for video_name, video_frame_images in all_video_data:
        video_save_path = os.path.join(args.output_dir, video_name)
        os.makedirs(video_save_path, exist_ok=True)

        input_image_batch = []
        input_image_gray_batch = []
        bname_batch = []
        
        # ========== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Сохраняем оригинальные размеры ==========
        # Создаем список для хранения оригинальных размеров каждого кадра
        original_sizes = []
        
        for image_name in video_frame_images:
            input_image = Image.open(image_name).convert('RGB')
            input_image_gray = input_image.convert('L')
            
            # Сохраняем оригинальные размеры ДО любых преобразований
            original_width, original_height = input_image.size
            original_sizes.append((original_width, original_height))
            
            rscale = args.upscale
            current_width, current_height = original_width, original_height

            # 1. Проверка минимального размера
            if original_width < args.process_size // rscale or original_height < args.process_size // rscale:
                scale = (args.process_size // rscale) / min(original_width, original_height)
                current_width = int(scale * original_width)
                current_height = int(scale * original_height)
                input_image = input_image.resize((current_width, current_height), Image.LANCZOS)
                input_image_gray = input_image_gray.resize((current_width, current_height), Image.LANCZOS)

            # 2. Апскейлинг
            current_width *= rscale
            current_height *= rscale
            input_image = input_image.resize((current_width, current_height), Image.LANCZOS)
            input_image_gray = input_image_gray.resize((current_width, current_height), Image.LANCZOS)

            # 3. Выравнивание до кратности 8
            current_width = current_width - (current_width % 8)
            current_height = current_height - (current_height % 8)
            input_image = input_image.resize((current_width, current_height), Image.LANCZOS)
            input_image_gray = input_image_gray.resize((current_width, current_height), Image.LANCZOS)

            bname = os.path.basename(image_name)
            bname_batch.append(bname)
            input_image_batch.append(input_image)
            input_image_gray_batch.append(input_image_gray)
        # ========== КОНЕЦ ИСПРАВЛЕНИЯ ==========

        exist_prompt = 0
        for input_image_index in range(0, len(input_image_batch), (frame_num - frame_overlap)):
            if input_image_index + frame_num - 1 >= len(input_image_batch):
                end = len(input_image_batch) - input_image_index
                start = 0
            else:
                start = 0
                end = frame_num

            input_frames = []
            input_frames_gray = []
            for input_frame_index in range(start, end):
                real_idx = input_image_index + input_frame_index
                if real_idx < 0 or real_idx >= len(input_image_batch):
                    continue

                current_frame = transforms.functional.to_tensor(input_image_batch[real_idx])
                current_frame_gray = transforms.functional.to_tensor(input_image_gray_batch[real_idx])
                current_frame_gray = torch.nn.functional.interpolate(current_frame_gray.unsqueeze(0), scale_factor=0.125).squeeze(0)
                input_frames.append(current_frame)
                input_frames_gray.append(current_frame_gray)

                # Получаем промпт только для первого кадра в батче
                if exist_prompt == 0:
                    validation_prompt = get_validation_prompt(args, input_image_batch[real_idx], DAPE)
                    if args.save_prompts:
                        txt_save_path = f"{txt_path}/{bname_batch[real_idx].split('.')[0]}.txt"
                        with open(txt_save_path, 'w', encoding='utf-8') as f:
                            f.write(validation_prompt)
                    exist_prompt = 1

            input_image_final = torch.stack(input_frames, dim=0)
            input_image_gray_final = torch.stack(input_frames_gray, dim=0)

            uncertainty_map = []
            if input_image_final.shape[0] > 1:
                for image_index in range(1, input_image_final.shape[0]):
                    compute_frame = torch.stack([
                        input_image_gray_final[image_index], 
                        input_image_gray_final[image_index - 1]
                    ])
                    uncertainty_map_each = compute_frame_difference_mask(compute_frame)
                    uncertainty_map.append(uncertainty_map_each)

            if uncertainty_map:
                uncertainty_map = torch.stack(uncertainty_map)
            else:
                uncertainty_map = torch.zeros_like(input_image_gray_final[0:1])

            with torch.no_grad():
                c_t = input_image_final.unsqueeze(0).cuda() * 2 - 1
                c_t = c_t.to(dtype=weight_dtype)
                output_image, _, _, _, _ = model(
                    stages=args.stages, 
                    c_t=c_t, 
                    uncertainty_map=uncertainty_map.unsqueeze(0).cuda(), 
                    prompt=validation_prompt, 
                    weight_dtype=weight_dtype
                )

            frame_t = output_image[0]
            frame_t = (frame_t.cpu() * 0.5 + 0.5)
            output_pil = transforms.ToPILImage()(frame_t)

            src_idx = input_image_index + start + 1
            if src_idx < 0 or src_idx >= len(input_image_batch):
                src_idx = max(0, min(src_idx, len(input_image_batch) - 1))

            source_pil = input_image_batch[src_idx]

            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=source_pil)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=source_pil)
            
            # ========== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Восстановление размера ==========
            # Всегда масштабируем к целевому размеру: original * upscale
            original_width, original_height = original_sizes[src_idx]
            target_width = original_width * args.upscale
            target_height = original_height * args.upscale
            
            # Только если текущий размер не совпадает с целевым
            if output_pil.size != (target_width, target_height):
                output_pil = output_pil.resize((target_width, target_height), Image.LANCZOS)
            # ========== КОНЕЦ ИСПРАВЛЕНИЯ ==========

            global_frame_counter = src_idx
            out_name = f"frame_{global_frame_counter:04d}.png"
            out_path = f"{video_save_path}/{out_name}"

            output_pil.save(out_path)

            gc.collect()
            torch.cuda.empty_cache()

