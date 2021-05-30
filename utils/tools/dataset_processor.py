# Author: Jintao Huang
# Time: 2020-5-24

import os
import numpy as np
import cv2 as cv
from ..display import imread, resize_max


class DatasetProcessor:
    """$"""

    def __init__(self, root_dir, labels=None):
        """

        :param root_dir: str
        :param labels: list[str] or Dict[str: int]  可多对一
        """
        self.root_dir = root_dir
        labels = labels or os.listdir(root_dir)
        if isinstance(labels, list):
            labels = dict(zip(labels, range(len(labels))))
        self.labels_str2int = labels
        self.labels_int2str = [k for k, v in labels.items() if v >= 0]
        self.image_path_list = None  # Len[N_图片]
        self.target_list = None  # Len[N_图片]

    def parse_dataset(self):
        self.image_path_list, self.target_list = [], []
        for label in self.labels_int2str:
            images_dir = os.path.join(self.root_dir, label)
            label = self.labels_str2int[label]
            for fname in os.listdir(images_dir):
                if fname.endswith('placeholder'): continue
                self.image_path_list.append(os.path.join(images_dir, fname))
                self.target_list.append(label)

    def test_dataset(self):
        """测试dataset文件(输出总图片数、各个分类的图片数).

        :return: None
        """
        print("-------------------------------------------------")
        print("images数量: %d" % len(self.image_path_list))
        print("targets数量: %d" % len(self.target_list))
        # 获取target各个类的数目
        # 1. 初始化classes_num_dict
        classes_num_dict = {label_name: 0 for label_name in self.labels_int2str}
        # 2. 累加
        for target in self.target_list:  # 遍历每一张图片
            classes_num_dict[self.labels_int2str[target]] += 1
        # 3. 打印
        print("classes_num:")
        for object_name, value in classes_num_dict.items():
            print("\t%s: %d" % (object_name, value))
        print("\tAll: %d" % sum(classes_num_dict.values()), flush=True)

    def show_dataset(self, random=False):
        """展示数据集，一张张展示

        :param random: bool
        :return: None
        """
        if random:
            orders = np.random.permutation(range(len(self.image_path_list)))
        else:
            orders = range(len(self.image_path_list))
        for i in orders:  # 随机打乱
            # 1. 数据结构
            img_path = self.image_path_list[i]
            target = self.target_list[i]
            # 2. 打开图片
            image = imread(img_path)
            image = resize_max(image, 720, 1080)
            cv_name = "%s_%s" % (os.path.split(img_path)[-1], self.labels_int2str[target])
            cv.imshow(cv_name, image)
            cv.waitKey(0)
            cv.destroyWindow(cv_name)
