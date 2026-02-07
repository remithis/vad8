# import os
# import json
# import re
# import torch
# from tqdm import tqdm
# from llm2vec import LLM2Vec
# from transformers import AutoTokenizer, LlamaModel, AutoConfig


# INPUT_JSON = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/ucf_crime_explanations.json"
# OUTPUT_EMB_DIR = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/ucf_video_embeddings"
# MODEL_NAME = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"

# os.makedirs(OUTPUT_EMB_DIR, exist_ok=True)

# def load_llm2clip_model():
#     """ 
#     使用均值池化聚合句子特征 
#     最大长度设为 512，支持长文本
#     """
#     print(f"正在加载 LLM2CLIP 兼容模型...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
#     # 使用标准 LlamaModel 加载，绕过 transformers 内部路径报错
#     llm_model = LlamaModel.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.bfloat16, 
#         device_map="auto",
#     )
    
#     model = LLM2Vec(
#         llm_model, 
#         tokenizer, 
#         pooling_mode="mean", 
#         max_length=512, 
#         doc_max_length=512
#     )
#     return model

# def main():
#     if not os.path.exists(INPUT_JSON):
#         print(f"❌ 错误：找不到文件 {INPUT_JSON}")
#         return

#     with open(INPUT_JSON, "r", encoding="utf-8") as f:
#         video_data = json.load(f)

#     model = load_llm2clip_model()
#     video_names = list(video_data.keys())
#     batch_size = 4  # Llama-3-8B 占用显存较大，建议设为 4-8 
    
#     print(f"开始提取 4096 维语义特征...")
#     for i in tqdm(range(0, len(video_names), batch_size)):
#         batch_keys = video_names[i : i + batch_size]
        
#         # 直接提取完整 explanation 句子
#         batch_texts = []
#         for k in batch_keys:
#             raw_text = video_data[k]["explanation"]
#             # 简单清洗：去除多余的换行和前后空格
#             clean_text = raw_text.replace('\n', ' ').strip()
#             batch_texts.append(clean_text)

#         with torch.no_grad():
#             # 提取文本嵌入向量 F_T
#             # 这里 LLM2Vec 会自动应用双向注意力建模 
#             embeddings = model.encode(batch_texts, convert_to_tensor=True)
        
#         for j, key in enumerate(batch_keys):
#             # 将键名（如 Fighting/Fighting005）转为安全文件名
#             safe_filename = key.replace("/", "_") + ".pt"
#             save_path = os.path.join(OUTPUT_EMB_DIR, safe_filename)
#             # 保存为 float32 的张量，作为 MADM 模块的文本引导特征 [cite: 569]
#             torch.save(embeddings[j].cpu().float(), save_path)

#     print(f"✅ 完成！向量文件已保存至: {OUTPUT_EMB_DIR}")

# if __name__ == "__main__":
#     main()

import os
import json
import torch
from tqdm import tqdm
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, LlamaModel

# ================= 1. 修改路径 =================
# 输入：你生成的 XD-Violence 解释文件
INPUT_JSON = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/xd_test_explanations.json"
# 输出：存放提取好的 4096 维特征向量
OUTPUT_EMB_DIR = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/xd_test_embeddings"
MODEL_NAME = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"

os.makedirs(OUTPUT_EMB_DIR, exist_ok=True)

def load_llm2clip_model():
    print(f"正在加载 LLM2CLIP 兼容模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 核心修改：移除 device_map="auto"，改用显式加载
    llm_model = LlamaModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16, 
        # device_map="auto",  <-- 删掉这一行或设为 None
        low_cpu_mem_usage=True, # 优化内存，但不使用 meta device
    ).to(device) # 直接将模型移至指定设备
    
    model = LLM2Vec(
        llm_model, 
        tokenizer, 
        pooling_mode="mean", 
        max_length=512, 
        doc_max_length=512
    )
    return model

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 错误：找不到文件 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    model = load_llm2clip_model()
    video_names = list(video_data.keys())
    
    # Llama-3-8B 占用显存较大，Batch Size 建议保持在 4-8
    batch_size = 4  
    
    print(f"开始提取 XD-Violence 语义特征向量...")
    for i in tqdm(range(0, len(video_names), batch_size)):
        batch_keys = video_names[i : i + batch_size]
        
        batch_texts = []
        for k in batch_keys:
            raw_text = video_data[k]["explanation"]
            # 清洗文本，移除干扰符号
            clean_text = raw_text.replace('\n', ' ').strip()
            batch_texts.append(clean_text)

        # 判断是否已经处理过，支持断点续传
        # 如果 batch 里所有的向量都已存在，则跳过提取
        all_exist = True
        for key in batch_keys:
            if not os.path.exists(os.path.join(OUTPUT_EMB_DIR, key.replace("/", "_") + ".pt")):
                all_exist = False
                break
        if all_exist: continue

        with torch.no_grad():
            # 提取 4096 维嵌入向量
            embeddings = model.encode(batch_texts, convert_to_tensor=True)
        
        for j, key in enumerate(batch_keys):
            # ================= 2. 修改文件名处理 =================
            # XD-Violence 的 key 已经是视频全名（无后缀），直接替换斜杠即可
            safe_filename = key.replace("/", "_") + ".pt"
            save_path = os.path.join(OUTPUT_EMB_DIR, safe_filename)
            
            # 保存为 float32 的张量，方便后续模型读取计算
            torch.save(embeddings[j].cpu().float(), save_path)

    print(f"✅ 完成！XD-Violence 向量文件已保存至: {OUTPUT_EMB_DIR}")

if __name__ == "__main__":
    main()