from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    # 加载预先提取的 7 个类别描述特征 [7, 4096]
    class_llm_feats = torch.load("xd_class_llm_feats.pt").to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        for i, item in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}")): # NEW
            step = 0
            visual_feat, clip_labels, feat_lengths, llm_video_feat= item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            llm_video_feat = llm_video_feat.to(device) # [B, 4096]
            text_labels = get_batch_label(clip_labels, prompt_text, label_map).to(device)

            # text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths) 

            # 1. 前向传播
            # fv: 帧级特征, t_norm: 注入后的文本特征, logits1: 帧分数, logits2: 视频-文本对齐分数
            fv, t_norm, logits1, logits2 = model(visual_feat, None, llm_video_feat, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()

            # loss2 = CLASM(logits2, text_labels, feat_lengths, device)

            # 计算 Loss2: 帧-类别对齐损失 (基于投影适配后的特征)
            # 将类别特征投影到视觉空间 [7, 512]
            T_class_all = model.text_proj(class_llm_feats) 
            T_class_all_norm = F.normalize(T_class_all, p=2, dim=-1)
            
            logit_scale = model.logit_scale.exp()
            v_norm = F.normalize(fv, p=2, dim=-1)
            
            # 计算帧与类别中心的相似度矩阵 [B, T, 7]
            frame_sims = torch.matmul(v_norm, T_class_all_norm.t()) 
            
            batch_alignment_logits = []
            for b in range(frame_sims.shape[0]):
                v_len = feat_lengths[b]
                video_sims = frame_sims[b, :v_len, :] # 排除 Padding 帧
                k = max(1, int(v_len / 32 + 1)) # Top-K 采样
                topk_sims, _ = torch.topk(video_sims, k=k, dim=0) 
                batch_alignment_logits.append(torch.mean(topk_sims, dim=0).unsqueeze(0))

            logits_alignment = torch.cat(batch_alignment_logits, dim=0) * logit_scale
            true_class_indices = torch.argmax(text_labels, dim=1)
            loss2 = F.cross_entropy(logits_alignment, true_class_indices)

            loss_total2 += loss2.item()

            # loss3 = torch.zeros(1).to(device)
            # text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            # for j in range(1, text_features.shape[0]):
            #     text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
            #     loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            # loss3 = loss3 / 6

            # 计算 Loss3: 类别语义分离损失 (Separation Loss)
            # 确保 Normal 特征和 Anomaly 特征被推开
            L_N = T_class_all_norm[0:1] # Normal
            L_A = T_class_all_norm[1:]  # 6个 Anomaly
            sim_na = torch.matmul(L_A, L_N.t()).squeeze()
            loss_push_norm = torch.sum(F.relu(sim_na + 0.4)) # Margin 控制

            # 推开 6 个异常类
            # sim_intra 形状为 [6, 6]
            sim_intra = torch.matmul(L_A, L_A.t())
            # 这里的 eye 参数要改成 6
            mask = torch.eye(6).to(device)
            # 惩罚非对角线上（不同类别间）相似度超过 0.3 的部分
            loss_dist = torch.mean(F.relu(sim_intra * (1 - mask) - 0.3))

            # 汇总 Loss3
            loss3 = loss_push_norm + loss_dist

            loss = loss1 + loss2 + loss3 * 1e-4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                
        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args.visual_length, class_llm_feats, gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})


    class_llm_feats = torch.load("xd_class_llm_feats.pt").to(device)
    llm_dir = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/xd_train_embeddings"
    llm_dir_test = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/xd_test_embeddings"

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, llm_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map, llm_dir_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    train(model, train_loader, test_loader, args, label_map, device)