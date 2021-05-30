# Author: Jintao Huang
# Time: 2020-5-21

from utils.tools import DatasetProcessor
from utils.utils import save_to_pickle, load_from_pickle
import os

# 数据集请在dataset_dir文件夹下按文件夹名进行分类
# (Please categorize the datasets by folder name under the folder dataset_dir)
# --------------------------------
dataset_dir = r'./dataset'  # 数据集所在文件夹
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
pkl_folder = 'pkl'  # 数据集pickle文件(缓存labels)
train_pickle_fname = "images_targets_train.pkl"
val_pickle_fname = "images_targets_val.pkl"
labels = [  # just for show
    "person", "car"
]
# -------------------------------- train_dataset
dataset_processor = DatasetProcessor(train_dir, labels)
dataset_processor.parse_dataset()
dataset_processor.test_dataset()
pkl_dir = os.path.join(dataset_dir, pkl_folder)
pkl_path = os.path.join(pkl_dir, train_pickle_fname)
os.makedirs(pkl_dir, exist_ok=True)
save_to_pickle((dataset_processor.image_path_list, dataset_processor.target_list), pkl_path)
# dataset_processor.show_dataset(random=False)
# -------------------------------- val_dataset
dataset_processor = DatasetProcessor(val_dir, labels)
dataset_processor.parse_dataset()
dataset_processor.test_dataset()
pkl_path = os.path.join(pkl_dir, val_pickle_fname)
save_to_pickle((dataset_processor.image_path_list, dataset_processor.target_list), pkl_path)
dataset_processor.show_dataset(random=False)
