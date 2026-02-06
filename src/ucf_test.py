import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option

def test(model, testdataloader, maxlen, class_llm_feats, gt, gtsegments, gtlabels, device):
    
    model.to(device)
    model.eval()

    element_logits2_stack = [] # 存储每个视频所有帧的分类概率


    with torch.no_grad():
        # 将 14 个类别的 4096 维 LLM 特征投影到 512 维空间并归一化
        # class_llm_feats 形状: [14, 4096]
        T_class = model.text_proj(class_llm_feats) 
        # 归一化
        T_class_norm = F.normalize(T_class, p=2, dim=-1) # [14, 512]

        # [DEBUG 1] 检查投影后的语义中心彼此是否太接近（如果太近，模型无法区分异常类别）
        sim_matrix = torch.matmul(T_class_norm, T_class_norm.t())
        print(f"[DEBUG] Class Center Similarity (Max self-sim should be 1.0):\n{sim_matrix[:3, :3]} (Top 3x3)")
        print(f"[DEBUG] Normal vs Anomaly mean sim: {sim_matrix[0, 1:].mean().item():.4f}")

        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0) # 提取视觉特征 [1, Time, 512] -> [Time, 512]
            length = item[2] # 视频总特征数

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0) # [B,T,512]

            visual = visual.to(device)

            # 将长视频切分为多个 chunk
            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            # 通过编码器获取经过时序建模后的视觉特征
            visual_features = model.encode_video(visual, padding_mask, lengths) 
            # 还原形状并截断 padding
            visual_features = visual_features.reshape(-1, visual_features.shape[-1])[0:len_cur]


            # _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            # 计算二分类异常置信度
            logits1 = model.classifier(visual_features + model.mlp2(visual_features))
            # 归一化到[0,1]
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            # logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            # logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            # prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            # 对帧级特征进行归一化
            visual_features_norm = F.normalize(visual_features, p=2, dim=-1)

            # === DEBUG 2: 检查视觉特征与语义特征是否对齐 ===
            if i == 0: # 仅对第一个视频输出
                print(f"[DEBUG] Visual Feats Norm mean: {visual_features_norm.mean().item():.4f}")
                raw_sim = torch.matmul(visual_features_norm, T_class_norm.t())
                print(f"[DEBUG] Raw Similarity Range: [{raw_sim.min().item():.4f}, {raw_sim.max().item():.4f}]")

            # 计算相似矩阵，帧级视觉特征 [len_cur, 512] @ 14个类别中心 [512, 14]
            logit_scale = model.logit_scale.exp()
            M_frame = (visual_features_norm @ T_class_norm.T) * logit_scale

            
            
            # 计算 A-Branch 的异常概率 prob2,对 14 个类别的相似度做 Softmax
            # (1 - 索引0的概率) 表示该帧不属于正常的概率，即异常概率
            prob2 = (1 - M_frame.softmax(dim=-1)[:, 0])

            # === DEBUG 3: 异常得分范围检查 ===
            if i % 100 == 0:
                print(f"[DEBUG] Video {i} | Scale: {logit_scale.item():.2f} | Prob1 Max: {prob1.max().item():.4f} | Prob2 Max: {prob2.max().item():.4f}")
            # 拼接所有视频的帧级预测分
            if i == 0:
                ap1 = prob1
                ap2 = prob2
                #ap3 = prob3
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            # element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            # element_logits2 = np.repeat(element_logits2, 16, 0)
            # element_logits2_stack.append(element_logits2)

            # 存储14类的概率，重复 16 次以匹配帧级 Ground Truth
            element_logits2 = M_frame.softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    # 加载离线 LLM 特征 [14, 4096]
    class_llm_feats = torch.load("class_llm_feats.pt").to(device)

    llm_dir = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/ucf_video_embeddings"
    testdataset = UCFDataset(args.visual_length, args.test_list, True, None, None, llm_dir)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    # prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, testdataloader, args.visual_length,class_llm_feats, gt, gtsegments, gtlabels, device)