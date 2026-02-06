# # import torch
# # import os
# # import glob
# # import numpy as np
# # from torch.nn.functional import cosine_similarity

# # # 配置路径
# # EMB_DIR = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/ucf_video_embeddings"

# # def check_embedding_quality():
# #     files = glob.glob(os.path.join(EMB_DIR, "*.pt"))
# #     if not files:
# #         print("❌ 错误：未在目录中找到 .pt 文件")
# #         return

# #     print(f"=== LLM2CLIP 特征检查报告 ===")
# #     print(f"1. 提取文件总数: {len(files)}")

# #     # 1. 维度与数值检查
# #     sample_data = torch.load(files[0])
# #     print(f"2. 特征维度: {sample_data.shape} (预期: [4096])") # 基于 Llama-3-8B [cite: 586]
    
# #     all_means = []
# #     all_stds = []
# #     for f in files[:100]: # 抽样100个进行统计
# #         data = torch.load(f).float()
# #         all_means.append(data.mean().item())
# #         all_stds.append(data.std().item())

# #     print(f"3. 数值稳定性: 平均值={np.mean(all_means):.6f}, 标准差={np.mean(all_stds):.6f}")
    
# #     # 2. 语义区分度检查 (余弦相似度)
# #     # 分别找两个同为 Normal 和两个不同类别的视频进行对比
# #     normal_files = [f for f in files if "Normal" in f][:2]
# #     anomaly_files = [f for f in files if "Normal" not in f][:1]

# #     if len(normal_files) >= 2 and len(anomaly_files) >= 1:
# #         f1 = torch.load(normal_files[0]).float().unsqueeze(0)
# #         f2 = torch.load(normal_files[1]).float().unsqueeze(0)
# #         f3 = torch.load(anomaly_files[0]).float().unsqueeze(0)

# #         sim_same = cosine_similarity(f1, f2).item()
# #         sim_diff = cosine_similarity(f1, f3).item()

# #         print(f"4. 语义相似度抽样:")
# #         print(f"   - 同类相似度 (Normal vs Normal): {sim_same:.4f}")
# #         print(f"   - 异类相似度 (Normal vs Anomaly): {sim_diff:.4f}")

# # check_embedding_quality()

# import torch
# import torch.nn.functional as F

# # 加载提取的类别特征 [14, 4096]
# FEAT_PATH = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/class_llm_feats.pt"
# class_feats = torch.load(FEAT_PATH)

# def check_class_embeddings(feats):
#     print("=== 类别特征 (Class Embeddings) 合理性检查 ===")
    
#     # 1. 维度检查
#     print(f"1. 特征形状: {feats.shape} (预期: [14, 4096])")
#     if feats.shape != (14, 4096):
#         print("   ❌ 警告: 维度不匹配！")

#     # 2. 数值有效性
#     has_nan = torch.isnan(feats).any()
#     print(f"2. 包含 NaN/Inf: {has_nan} (预期: False)")

#     # 3. 语义区分度 (余弦相似度分析)
#     # 归一化后计算相似度矩阵 [14, 14]
#     feats_norm = F.normalize(feats.float(), p=2, dim=-1)
#     sim_matrix = torch.mm(feats_norm, feats_norm.t())

#     # 提取 Normal (索引0) 与其他 Anomaly (索引1-13) 的相似度
#     normal_vs_anomalies = sim_matrix[0, 1:]
    
#     print(f"3. 语义距离分析:")
#     print(f"   - Normal 与各 Anomaly 的平均相似度: {normal_vs_anomalies.mean().item():.4f}")
#     print(f"   - 最大相似度: {normal_vs_anomalies.max().item():.4f} (对应的索引: {normal_vs_anomalies.argmax().item() + 1})")
#     print(f"   - 最小相似度: {normal_vs_anomalies.min().item():.4f}")

#     # 4. 类别内部相似度 (Anomaly 相互之间)
#     anomaly_sims = sim_matrix[1:, 1:]
#     # 排除对角线自身相似度
#     mask = torch.eye(13).bool()
#     avg_anomaly_sim = anomaly_sims[~mask].mean().item()
#     print(f"4. Anomaly 相互间的平均相似度: {avg_anomaly_sim:.4f}")

#     # 5. 合理性判定说明
#     print("\n判定标准提示:")
#     if normal_vs_anomalies.mean() > 0.8:
#         print("   ⚠️ 风险: Normal 与 Anomaly 太像了，Loss3 的负担会很重。")
#     elif avg_anomaly_sim > 0.9:
#         print("   ⚠️ 风险: 异常类别之间缺乏区分度。")
#     else:
#         print("   ✅ 正常: 特征空间具有初始区分度，适合进行适配器训练。")

# check_class_embeddings(class_feats)

import torch
import torch.nn.functional as F

# 1. 加载你的特征文件
file_path = "class_llm_feats.pt"
class_feats = torch.load(file_path) # 预期形状 [14, 4096]

# 2. 定义你应该有的标签顺序 (UCF-Crime 标准顺序)
labels = [
    'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
    'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

print(f"特征矩阵形状: {class_feats.shape}")

# 3. 计算类间相似度矩阵
class_feats_norm = F.normalize(class_feats, p=2, dim=-1)
sim_matrix = torch.mm(class_feats_norm, class_feats_norm.t())

print("\n--- 类别顺序检查 ---")
for i in range(len(labels)):
    # 检查当前类别与其之后类别的相似度
    # 如果 Normal (0) 与其他类的相似度极高 (>0.9)，说明语义区分度不足
    max_sim_idx = torch.argmax(sim_matrix[i, :])
    print(f"索引 {i:2d} | 预定标签: {labels[i]:15s} | 自我相似度: {sim_matrix[i,i]:.4f}")

# 4. 特别检查 Normal 与 Anomaly 的区分度
normal_feat = class_feats_norm[0]
anomaly_feats = class_feats_norm[1:]
mean_sim = torch.dot(normal_feat, anomaly_feats.mean(dim=0))
print(f"\n[结果] Normal 与其余异常类的平均相似度: {mean_sim.item():.4f}")
if mean_sim > 0.8:
    print("警告：正常类与异常类在 LLM 原始空间中太近了，模型很难推开它们！")