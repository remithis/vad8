import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

# class UCFDataset(data.Dataset):
#     def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
#         self.df = pd.read_csv(file_path)
#         self.clip_dim = clip_dim
#         self.test_mode = test_mode
#         self.label_map = label_map
#         self.normal = normal
#         if normal == True and test_mode == False:
#             self.df = self.df.loc[self.df['label'] == 'Normal']
#             self.df = self.df.reset_index()
#         elif test_mode == False:
#             self.df = self.df.loc[self.df['label'] != 'Normal']
#             self.df = self.df.reset_index()
        
#     def __len__(self):
#         return self.df.shape[0]

#     def __getitem__(self, index):
#         clip_feature = np.load(self.df.loc[index]['path'])
#         if self.test_mode == False:
#             clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
#         else:
#             clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

#         clip_feature = torch.tensor(clip_feature)
#         clip_label = self.df.loc[index]['label']
#         return clip_feature, clip_label, clip_length

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, 
                 normal: bool = False, llm_dir: str = None):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.llm_dir = llm_dir # 新增：4096维文本特征存放目录

        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # 1. 加载视觉特征
        video_path = self.df.loc[index]['path']
        clip_feature = np.load(video_path)
        
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        # 2. 加载对应的 LLM 4096维文本特征
        # 根据您之前的保存逻辑：key.replace("/", "_") + ".pt"
        # 假设 csv 中的 path 包含 'Fighting/Fighting005_x264.npy' 这种结构
        video_id = "/".join(video_path.split('/')[-2:]).replace('.npy', '')
        llm_feat_name = video_id.replace("/", "_") + ".pt"
        llm_feat_path = os.path.join(self.llm_dir, llm_feat_name)
        
        # 加载离线特征并转为 float32
        if os.path.exists(llm_feat_path):
            llm_text_feat = torch.load(llm_feat_path).float() # [4096]
        else:
            # 如果没找到（如Normal视频），返回全零向量作为占位
            llm_text_feat = torch.zeros(4096)

        return clip_feature, clip_label, clip_length, llm_text_feat

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, llm_dir: str = None):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.llm_dir = llm_dir
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # clip_feature = np.load(self.df.loc[index]['path'])
        video_path = self.df.loc[index]['path']
        clip_feature = np.load(video_path)

        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        # 加载对应的离线 LLM 文本特征 (4096维)
        # 假设 csv 中 path 是 '/path/to/video_name.npy'，我们需要 video_name
        # video_id = os.path.basename(video_path).replace('.npy', '')
        # llm_feat_path = os.path.join(self.llm_dir, video_id + ".pt")
        
        # if self.llm_dir is not None and os.path.exists(llm_feat_path):
        #     # 加载并转为 float32
        #     llm_text_feat = torch.load(llm_feat_path).float()
        # else:
        #     # 如果是测试模式或未找到特征，返回零向量
        #     llm_text_feat = torch.zeros(4096)
        llm_text_feat = torch.zeros(4096)  # 默认初始化为零向量
        
        if self.llm_dir is not None:
            video_id = os.path.basename(video_path).replace('.npy', '')
            llm_feat_path = os.path.join(self.llm_dir, video_id + ".pt")
            
            if os.path.exists(llm_feat_path):
                llm_text_feat = torch.load(llm_feat_path).float()
            else:
                # 打印一个警告，方便调试缺少的特征
                # print(f"Warning: LLM feature not found at {llm_feat_path}")
                pass

        return clip_feature, clip_label, clip_length, llm_text_feat