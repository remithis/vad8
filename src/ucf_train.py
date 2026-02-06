import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

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
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
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
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            # normal_features, normal_label, normal_lengths = next(normal_iter)
            # anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)
            # 获取数据，现在包含第四项 llm_feat
            normal_features, normal_label, normal_lengths, normal_llm_feat = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_llm_feat = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)

            # 拼接离线提取的 4096 维 LLM 文本特征
            llm_text_features = torch.cat([normal_llm_feat, anomaly_llm_feat], dim=0).to(device)

            # text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            # text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_labels_list = list(normal_label) + list(anomaly_label)
            text_labels = get_batch_label(text_labels_list, prompt_text, label_map).to(device)

            fv, t_norm, logits1, logits2 = model(visual_features, None, llm_text_features, feat_lengths) 


            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            # #loss2
            # loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            # loss_total2 += loss2.item()

            ##计算loss2

            # T_class_all = model.text_proj(class_llm_feats) 
            # T_class_all_norm = F.normalize(T_class_all, p=2, dim=-1) # [14, 512]

            # true_class_indices = torch.argmax(text_labels, dim=1).to(device)

            # # 计算视频全局特征 v_norm 与 14 个类别中心的相似度
            # # v_norm 是 model(visual_features, ...) 返回的第一个值
            # logit_scale = model.logit_scale.exp()
            # logits_alignment = torch.matmul(t_norm, T_class_all_norm.t()) * logit_scale

            # # 使用 CrossEntropy 让视频特征向正确的类别中心对齐
            # loss2 = F.cross_entropy(logits_alignment, true_class_indices)
            # loss_total2 += loss2.item()

            # --- train.py 内部修改 ---

            # 1. 投影类别中心 [14, 512]
            T_class_all = model.text_proj(class_llm_feats) 
            T_class_all_norm = F.normalize(T_class_all, p=2, dim=-1)

            # 2. 这里的 visual_features 是 encode_video 后的帧级特征 [Batch_total, 512]
            # 我们需要按视频拆分回 [Batch_size, Time, 512]
            # 注意：这里需要确保 visual_features 已经根据 lengths 还原并对齐
            # 假设 visual_features 形状为 [B, T, 512] (通常在 model forward 里处理)

            # 计算每一帧与 14 个类别的相似度 -> [B, T, 14]
            # 使用 model.logit_scale.exp()
            logit_scale = model.logit_scale.exp()
            v_norm = F.normalize(fv, p=2, dim=-1)
            frame_sims = torch.matmul(v_norm, T_class_all_norm.t()) # [B, T, 14]

            # 3. 对每个视频，针对每个类别选取 Top-K 帧
            # K 的取值通常设为 T/16 + 1
            batch_alignment_logits = []
            for b in range(frame_sims.shape[0]):
                v_len = feat_lengths[b]
                # 取该视频的前 v_len 帧，避免 padding 干扰
                video_sims = frame_sims[b, :v_len, :] # [T_actual, 14]
                
                k = max(1, int(v_len / 16 + 1))
                # 在时间维度 (dim=0) 选出每类得分最高的前 k 个
                topk_sims, _ = torch.topk(video_sims, k=k, dim=0) # [k, 14]
                
                # 计算均值作为该视频对 14 类的最终得分
                video_logits = torch.mean(topk_sims, dim=0) # [14]
                batch_alignment_logits.append(video_logits.unsqueeze(0))

            # 4. 拼接成 Batch 结果 [B, 14] 并缩放
            logits_alignment = torch.cat(batch_alignment_logits, dim=0) * logit_scale

            # 5. 计算 CrossEntropy
            true_class_indices = torch.argmax(text_labels, dim=1).to(device)
            loss2 = F.cross_entropy(logits_alignment, true_class_indices)

            # # 获取当前 Batch 的样本数量 (Batch_size * 2, 因为包含了 normal 和 anomaly)
            # batch_size_current = logits2.shape[0]

            # # 构造对比学习的 Ground Truth 标签
            # # 在 logits2 [B, B] 矩阵中，对角线上的元素 (i, i) 代表视频 i 和它自己的文本 i 匹配
            # # 因此标签就是从 0 到 B-1 的序列
            # contrastive_labels = torch.arange(batch_size_current).long().to(device)

            # # 计算视频对齐到文本的损失
            # # 每一行代表一个视频在 Batch 内所有文本中的相似度分布
            # loss_v2t = F.cross_entropy(logits2, contrastive_labels)

            # # 计算文本对齐到视频的损失
            # # 每一列代表一个文本在 Batch 内所有视频中的相似度分布
            # # 对 logits2 进行转置，使得每一行代表一个文本对所有视频的相似度
            # loss_t2v = F.cross_entropy(logits2.T, contrastive_labels)

            # # 取双向损失的平均值作为最终的 loss2 (对齐损失)
            # loss2 = (loss_v2t + loss_t2v) / 2.0

            loss_total2 += loss2.item()

            #loss3 
            # loss3 = torch.zeros(1).to(device)
            # text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            # for j in range(1, text_features.shape[0]):
            #     text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
            #     loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            # loss3 = loss3 / 13 * 1e-1



            L_N = T_class_all_norm[0:1] # Normal 中心
            L_A = T_class_all_norm[1:]  # 13个 Anomaly 中心

            # (A) 推开 Normal 和 Anomaly: Margin 0.5
            sim_na = torch.matmul(L_A, L_N.t()).squeeze()
            loss_push_norm = torch.sum(F.relu(sim_na + 0.4)) 

            # (B) 推开 13 个异常类: 提高辨识度
            sim_intra = torch.matmul(L_A, L_A.t())
            mask = torch.eye(13).to(device)
            loss_dist = torch.mean(F.relu(sim_intra * (1 - mask) - 0.3))

            
            loss3 = loss_push_norm + loss_dist


            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(model, testloader, args.visual_length, class_llm_feats, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        
        torch.save(model.state_dict(), 'model/model_cur.pth')
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
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    llm_dir = "/home/xuchen/Project/VadCLIP-main-yxl/VadCLIP-new/ucf_video_embeddings" # new
    class_llm_feats = torch.load("class_llm_feats.pt").to(device)
    
    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True, llm_dir)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False, llm_dir)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map, None, llm_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)