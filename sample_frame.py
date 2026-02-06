import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path

# 原始数据集根目录
DATA_ROOT = "/home/xuchen/Data/UCF-Crime/"
# 采样帧保存目录
OUTPUT_ROOT = "./ucf_crime_sampled_frames/"

# 均匀采样32帧
SAMPLE_NUM = 32  

USE_INTERVAL = False
INTERVAL = 16 

def extract_frames_uniformly(video_path, save_dir, num_frames):

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Warning: Could not read frames from {video_path}")
        return

    # 计算采样索引
    if USE_INTERVAL:
        indices = range(0, total_frames, INTERVAL)
    else:
        # 使用 np.linspace 在 0 到 total_frames-1 之间均匀取 num_frames 个点
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        # 去重并排序
        indices = sorted(list(set(indices)))

    # 读取并保存帧
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 文件名格式: frame_帧序号.jpg
            save_name = os.path.join(save_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(save_name, frame)
    
    cap.release()

def main():
    # 查找所有视频文件
    video_files = []
    video_files.extend(glob.glob(os.path.join(DATA_ROOT, "**/*.mp4"), recursive=True))
    video_files.extend(glob.glob(os.path.join(DATA_ROOT, "**/*.avi"), recursive=True))
    
    print(f"Found {len(video_files)} videos in {DATA_ROOT}")
    
    if len(video_files) == 0:
        print("Error: No videos found. Please check the path.")
        return

    # 遍历
    for video_path in tqdm(video_files, desc="Sampling Frames"):
        video_path = Path(video_path)
        
        # 获取视频的相对路径
        try:
            relative_path = video_path.relative_to(DATA_ROOT)
        except ValueError:
            # 如果路径不匹配，直接使用文件名作为子文件夹
            relative_path = Path(video_path.parent.name) / video_path.name

        # 构建该视频对应的帧保存文件夹
        video_name_stem = video_path.stem # 去掉扩展名
        category_name = video_path.parent.name
        
        save_dir = os.path.join(OUTPUT_ROOT, category_name, video_name_stem)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            extract_frames_uniformly(video_path, save_dir, SAMPLE_NUM)
        else:
            # 如果文件夹已存在且不为空，跳过 (断点续传)
            if len(os.listdir(save_dir)) > 0:
                continue
            else:
                extract_frames_uniformly(video_path, save_dir, SAMPLE_NUM)

    print(f"Done! All frames saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()