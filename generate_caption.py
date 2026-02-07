# import os
# import json
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from transformers import CLIPProcessor, CLIPModel


# FRAMES_ROOT = "./ucf_crime_sampled_frames/"
# OUTPUT_JSON = "./ucf_crime_captions.json"
# BLIP_MODEL_ID = "Salesforce/blip2-opt-2.7b" 
# CLIP_MODEL_ID = "openai/clip-vit-base-patch16" 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# BATCH_SIZE = 4  # 根据显存调整 BLIP-2 的批处理大小
# NUM_BEAMS = 5   # 增加 Beam Search 提高生成质量


# def load_models():
#     print(f"Loading models to {DEVICE}...")
#     # 加载 BLIP-2
#     blip_processor = Blip2Processor.from_pretrained(BLIP_MODEL_ID)
#     blip_model = Blip2ForConditionalGeneration.from_pretrained(
#         BLIP_MODEL_ID, torch_dtype=torch.float16
#     ).to(DEVICE)
    
#     # 加载 CLIP
#     clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
#     clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    
#     return blip_processor, blip_model, clip_processor, clip_model

# @torch.no_grad()
# @torch.no_grad()
# def generate_captions_batched(blip_processor, blip_model, image_paths, batch_size=4):
#     """
#     修改后的生成函数：
#     1. 使用 Q-A 模板引导模型生成 
#     2. 只保留 Answer 之后的内容
#     """
#     # 论文中指定的 Prompt
#     original_prompt = "Please use short words to describe what is happening in the picture."
    
#     # 构造引导模板，强制模型进入回答模式
#     qa_prompt = f"Question: {original_prompt} Answer:"
#     all_captions = []
    
#     for i in range(0, len(image_paths), batch_size):
#         batch_paths = image_paths[i : i + batch_size]
#         images = [Image.open(p).convert("RGB") for p in batch_paths]
        
#         # 批量处理输入
#         inputs = blip_processor(
#             images=images, 
#             text=[qa_prompt] * len(images), 
#             return_tensors="pt"
#         ).to(DEVICE, torch.float16)
        
#         # 生成配置
#         out = blip_model.generate(
#             **inputs, 
#             max_new_tokens=50, 
#             num_beams=NUM_BEAMS, 
#             min_length=5,
#             do_sample=False  # 关闭随机采样以获得更稳定的描述
#         )
        
#         # 解码并清洗：核心修改点
#         # skip_special_tokens=True 移除标记
#         # 然后手动移除 prompt 部分的内容，确保只留下 answer 
#         captions = blip_processor.batch_decode(out, skip_special_tokens=True)
        
#         for caption in captions:
#             # 有时模型会把 Question 和 Answer 标签也吐出来，我们只需要最后一部分
#             clean_text = caption.split("Answer:")[-1].strip()
#             # 如果模型没吐出 Answer 标签但复读了 Question，则移除 Question 部分
#             clean_text = clean_text.replace(original_prompt, "").strip()
#             all_captions.append(clean_text)
        
#     return all_captions

# @torch.no_grad()

# @torch.no_grad()
# def clean_captions_efficient(clip_processor, clip_model, image_paths, raw_captions):
#     """
#     实现 Ex-VAD 论文中的图像-文本对齐清洗机制 [cite: 7, 58, 173]
#     T_i = arg max {cosine_similarity(E_I(I_i), E_T(T))} [cite: 176]
#     """
#     unique_texts = list(set(raw_captions))
    
#     # 预先编码所有候选文本的特征 (E_T) [cite: 179]
#     # text_inputs = clip_processor(text=unique_texts, return_tensors="pt", padding=True).to(DEVICE)
    
#     text_inputs = clip_processor(
#         text=unique_texts, 
#         return_tensors="pt", 
#         padding=True, 
#         truncation=True,    # 开启截断
#         max_length=77       # 强制限制在 CLIP 的 77 Token 范围内
#     ).to(DEVICE)
#     text_outputs = clip_model.get_text_features(**text_inputs)
    
#     # 如果返回的是对象，提取 pooler_output 属性 [cite: 54]
#     if hasattr(text_outputs, "pooler_output"):
#         text_features = text_outputs.pooler_output
#     else:
#         text_features = text_outputs # 已经是 Tensor
        
#     text_features = text_features.detach()
#     text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

#     cleaned_results = []
    
#     # 逐帧提取图像特征并匹配 (E_I) 
#     for i, img_path in enumerate(image_paths):
#         image = Image.open(img_path).convert("RGB")
#         image_inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        
#         image_outputs = clip_model.get_image_features(**image_inputs)
        
#         # 同上，提取图像侧的 pooler_output
#         if hasattr(image_outputs, "pooler_output"):
#             image_features = image_outputs.pooler_output
#         else:
#             image_features = image_outputs
            
#         image_features = image_features.detach()
#         image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
#         # 计算余弦相似度矩阵
#         similarities = (image_features @ text_features.T)
#         best_idx = similarities.argmax().item()
        
#         cleaned_results.append({
#             "frame_path": img_path,
#             "raw_caption": raw_captions[i],
#             "clean_caption": unique_texts[best_idx]
#         })
        
#     return cleaned_results

# def main():
#     blip_proc, blip_model, clip_proc, clip_model = load_models()
    
#     video_dirs = []
#     for root, dirs, files in os.walk(FRAMES_ROOT):
#         if not dirs and files:
#             video_dirs.append(root)
            
#     all_results = {}

#     for video_dir in tqdm(video_dirs, desc="Processing Video Captions"):

#         video_rel_path = os.path.relpath(video_dir, FRAMES_ROOT)
        
#         # 断点续传逻辑：如果该视频已在结果中，跳过
#         if video_rel_path in all_results:
#             continue
        
        
#         img_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".jpg")])
#         if not img_paths: continue
        
#         # 批量生成初始 Captions
#         raw_captions = generate_captions_batched(blip_proc, blip_model, img_paths, batch_size=BATCH_SIZE)
        
#         # 高效对齐清洗
#         cleaned_data = clean_captions_efficient(clip_proc, clip_model, img_paths, raw_captions)
        
#         video_rel_path = os.path.relpath(video_dir, FRAMES_ROOT)
#         all_results[video_rel_path] = cleaned_data
        
#         with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#             json.dump(all_results, f, indent=4, ensure_ascii=False)

#     print(f"Done! Captions saved to {OUTPUT_JSON}")

# if __name__ == "__main__":
#     main()

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

# --- 1. 修改路径 ---
FRAMES_ROOT = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/xd_test_sampled_frames"
OUTPUT_JSON = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/xd_test_captions.json"
BLIP_MODEL_ID = "Salesforce/blip2-opt-2.7b" 
CLIP_MODEL_ID = "openai/clip-vit-base-patch16" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8  # 如果显存大，可以调高到 8 或 16
NUM_BEAMS = 5 

def load_models():
    print(f"Loading models to {DEVICE}...")
    # BLIP-2 加载
    blip_processor = Blip2Processor.from_pretrained(BLIP_MODEL_ID)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP_MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)
    
    # CLIP 加载
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    
    return blip_processor, blip_model, clip_processor, clip_model


@torch.no_grad()
@torch.no_grad()
def generate_captions_batched(blip_processor, blip_model, image_paths, batch_size=4):
    """
    修改后的生成函数：
    1. 使用 Q-A 模板引导模型生成 
    2. 只保留 Answer 之后的内容
    """
    # 论文中指定的 Prompt
    original_prompt = "Please use short words to describe what is happening in the picture."
    
    # 构造引导模板，强制模型进入回答模式
    qa_prompt = f"Question: {original_prompt} Answer:"
    all_captions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        # 批量处理输入
        inputs = blip_processor(
            images=images, 
            text=[qa_prompt] * len(images), 
            return_tensors="pt"
        ).to(DEVICE, torch.float16)
        
        # 生成配置
        out = blip_model.generate(
            **inputs, 
            max_new_tokens=50, 
            num_beams=NUM_BEAMS, 
            min_length=5,
            do_sample=False  # 关闭随机采样以获得更稳定的描述
        )
        
        # 解码并清洗：核心修改点
        # skip_special_tokens=True 移除标记
        # 然后手动移除 prompt 部分的内容，确保只留下 answer 
        captions = blip_processor.batch_decode(out, skip_special_tokens=True)
        
        for caption in captions:
            # 有时模型会把 Question 和 Answer 标签也吐出来，我们只需要最后一部分
            clean_text = caption.split("Answer:")[-1].strip()
            # 如果模型没吐出 Answer 标签但复读了 Question，则移除 Question 部分
            clean_text = clean_text.replace(original_prompt, "").strip()
            all_captions.append(clean_text)
        
    return all_captions

@torch.no_grad()

@torch.no_grad()
def clean_captions_efficient(clip_processor, clip_model, image_paths, raw_captions):
    """
    实现 Ex-VAD 论文中的图像-文本对齐清洗机制 [cite: 7, 58, 173]
    T_i = arg max {cosine_similarity(E_I(I_i), E_T(T))} [cite: 176]
    """
    unique_texts = list(set(raw_captions))
    
    # 预先编码所有候选文本的特征 (E_T) [cite: 179]
    # text_inputs = clip_processor(text=unique_texts, return_tensors="pt", padding=True).to(DEVICE)
    
    text_inputs = clip_processor(
        text=unique_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,    # 开启截断
        max_length=77       # 强制限制在 CLIP 的 77 Token 范围内
    ).to(DEVICE)
    text_outputs = clip_model.get_text_features(**text_inputs)
    
    # 如果返回的是对象，提取 pooler_output 属性 [cite: 54]
    if hasattr(text_outputs, "pooler_output"):
        text_features = text_outputs.pooler_output
    else:
        text_features = text_outputs # 已经是 Tensor
        
    text_features = text_features.detach()
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    cleaned_results = []
    
    # 逐帧提取图像特征并匹配 (E_I) 
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        image_inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        
        image_outputs = clip_model.get_image_features(**image_inputs)
        
        # 同上，提取图像侧的 pooler_output
        if hasattr(image_outputs, "pooler_output"):
            image_features = image_outputs.pooler_output
        else:
            image_features = image_outputs
            
        image_features = image_features.detach()
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # 计算余弦相似度矩阵
        similarities = (image_features @ text_features.T)
        best_idx = similarities.argmax().item()
        
        cleaned_results.append({
            "frame_path": img_path,
            "raw_caption": raw_captions[i],
            "clean_caption": unique_texts[best_idx]
        })
        
    return cleaned_results

def main():
    # --- 2. 改进的断点续传逻辑 ---
    all_results = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
                print(f"检测到已存在的 JSON，已跳过 {len(all_results)} 个视频。")
            except:
                all_results = {}

    blip_proc, blip_model, clip_proc, clip_model = load_models()
    
    # 查找所有包含图片的子文件夹
    video_dirs = []
    for folder in os.listdir(FRAMES_ROOT):
        full_path = os.path.join(FRAMES_ROOT, folder)
        if os.path.isdir(full_path):
            video_dirs.append(full_path)
            
    # 排序以保证处理顺序一致
    video_dirs.sort()

    for video_dir in tqdm(video_dirs, desc="Generating Captions"):
        video_rel_path = os.path.basename(video_dir) # 使用文件夹名作为 Key
        
        # 跳过已处理的视频
        if video_rel_path in all_results:
            continue
        
        img_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".jpg")])
        if not img_paths: 
            continue
        
        try:
            # 1. 批量生成初始 Captions
            raw_captions = generate_captions_batched(blip_proc, blip_model, img_paths, batch_size=BATCH_SIZE)
            
            # 2. CLIP 对齐清洗
            cleaned_data = clean_captions_efficient(clip_proc, clip_model, img_paths, raw_captions)
            
            # 3. 写入结果
            all_results[video_rel_path] = cleaned_data
            
            # 每处理完一个视频就保存一次，防止程序崩溃丢失进度
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"处理视频 {video_rel_path} 时出错: {e}")
            continue

    print(f"✨ 处理完成! 描述文件保存在: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()