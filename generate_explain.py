import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置参数 =================
INPUT_JSON = "./ucf_crime_captions.json"
OUTPUT_EXPLAIN_JSON = "./ucf_crime_explanations.json"

# 使用官方 transformers 兼容的模型路径
MODEL_NAME = "unsloth/llama-3-8b-instruct-bnb-4bit" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def load_llama_standard():
    print(f"Loading Llama-3 via Transformers on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_video_explanation(model, tokenizer, captions_list):
    # 1. 时序聚合
    aggregated_text = " ".join([f"Frame {i}: {c['clean_caption']}" for i, c in enumerate(captions_list)])

    # 2. 优化后的 Prompt
    # 增加强制结尾格式 Conclusion: Yes/No，便于稳定提取 is_anomaly
    # 建议的 Prompt 片段
    crime_categories = "Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism"


    prompt = (
        f"Contextual Frames: {aggregated_text}\n\n"
        "Role: Expert Security Analyst.\n"
        "Task: Determine if the video is abnormal based on these categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.\n"
        "Constraint: Your response MUST be a single concise sentence. DO NOT mention frame numbers. DO NOT say 'The video shows'.\n"
        "Format: [Yes/No]. [Short summary of the specific event and its cause].\n"
        "Example: Yes. In an office, a distressed person is restrained by officers after control is gained."
    )

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(DEVICE)

    # 3. 生成解释 E
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300, # 稍大一点以容纳原因分析
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return response

def main():
    model, tokenizer = load_llama_standard()
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
        
    all_explanations = {}
    
    for video_rel_path, frames in tqdm(caption_data.items(), desc="Generating Explanations"):
        # 在 main 函数循环中
        full_response = generate_video_explanation(model, tokenizer, frames)

        # 模拟论文格式提取：检查开头是否为 Yes
        is_anomaly = full_response.startswith("Yes")

        all_explanations[video_rel_path] = {
            "explanation": full_response, # 这样保存的就是类似 "Yes. A man is..." 的纯净文本
            "is_anomaly": is_anomaly
        }
        
            
        # 实时保存，防止长耗时任务中断
        if len(all_explanations) % 10 == 0:
            with open(OUTPUT_EXPLAIN_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_explanations, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_EXPLAIN_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_explanations, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()