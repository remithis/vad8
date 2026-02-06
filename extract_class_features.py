import torch
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, LlamaModel

# ================= 配置 =================
MODEL_NAME = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
OUTPUT_PATH = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/class_llm_feats.pt"

# 严格按照你 train.py 中的顺序排列
CLASS_DESCRIPTIONS = [
    "The video shows normal daily activities without any suspicious or violent behavior.", # Normal
    "A person is being physically or verbally mistreated or harmed by another individual.", # Abuse
    "Police officers or security personnel are taking a suspect into custody.",              # Arrest
    "A person is intentionally setting fire to property, buildings, or vehicles.",          # Arson
    "An individual is physically attacking or striking another person with force.",         # Assault
    "Someone is illegally breaking into a building or house to commit theft.",              # Burglary
    "A sudden and violent release of energy causes a blast, fire, and destruction.",        # Explosion
    "Two or more individuals are engaged in a physical struggle or violent altercation.",   # Fighting
    "A vehicle collision or crash occurs on the road, involving cars or pedestrians.",      # RoadAccidents
    "A person is stealing property from another by using force, threats, or weapons.",      # Robbery
    "A person is discharging a firearm or gun at targets or other individuals.",           # Shooting
    "Someone is surreptitiously stealing goods from a retail store.",                       # Shoplifting
    "An individual is taking someone else's property without permission.",                 # Stealing
    "A person is intentionally damaging or defacing public or private property."            # Vandalism
]

def main():
    print("正在加载 LLM2CLIP 模型进行类别特征提取...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model = LlamaModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # 开启双向注意力和均值池化
    model = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512)

    with torch.no_grad():
        # 提取特征 [14, 4096]
        print(f"开始对 14 个类别进行语义编码...")
        embeddings = model.encode(CLASS_DESCRIPTIONS, convert_to_tensor=True)
        
    # 保存为 float32 的 pt 文件
    torch.save(embeddings.cpu().float(), OUTPUT_PATH)
    print(f"✅ 类别特征已保存至: {OUTPUT_PATH}")
    print(f"特征形状: {embeddings.shape}")

if __name__ == "__main__":
    main()