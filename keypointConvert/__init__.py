#  Copyright (c) 2023. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: __init__.py
@Description: 使用MS-G3D数据处理方法处理野生视频提取出的关节点数据
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/9/8 16:01 at PyCharm
"""

import os
import pickle
import re
import sys

import pandas as pd
from numpy.lib.format import open_memmap

import numpy as np
sys.path.append("./keypointConvert")
from preprocess import pre_normalization
from gen_bone_data import process_bone


class Process():
    def __init__(self, data_path, out_path, csv_name="pose-data.csv"):
        self.data_path = data_path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        self.val_number = [1]
        self.out_path = out_path
        self.max_joint = 17
        self.max_body = 2
        self.max_frame = 300
        self.image_w = 1920
        self.image_h = 1080
        self.label_test = {"test": 0}
        self.csv_name = csv_name

    def load_joint_data(self):
        labels = []
        files = os.listdir(self.data_path)
        sorted(files)
        print(files)
        for f in files:
            labels.append(self.label_test['test'])
        print("process infer data...")
        self.sub_process("infer", files, labels, csv_name=self.csv_name)

    def sub_process(self, part: str, sample_name, sample_label, csv_name="pose-data.csv"):
        """
        :param sample_label:
        :param sample_name:
        :param part: "val" or "train"
        :return:
        """
        with open(os.path.join(f'{self.out_path}', f'{part}_label.pkl'), 'wb') as f:
            pickle.dump((sample_name, sample_label), f)

        fp = open_memmap(
            os.path.join(f'{self.out_path}', f'{part}_data_joint.npy'),
            dtype='float32',
            mode='w+',
            shape=(len(sample_label), 3, self.max_frame, self.max_joint, self.max_body))

        for i, s in enumerate(sample_name):
            data = self.read_xy_unified_padding_z_with_zero(os.path.join(self.data_path, s, csv_name))
            fp[i, :, 0:data.shape[1], :, :] = data

        # there are three processes in pre_normalization
        # 1. padding the null frame with previous frames
        # 2. every point subtract body center: adjust origin point
        # 3. modify body to initial places like, spine parallels to z-axis and shoulders parallels to x-axis
        #   5 left shoulder, 6 right shoulder 5-6 should parallel to x-axis
        #   the middle of #5 and #6 are neck, while the middle of #11 and #12 are spine
        #   line between two middle points is spine, which should be parallel to z-axis
        fp = pre_normalization(fp, zaxis=[5, 6, 11, 12], xaxis=[5, 6])

    def read_xy_unified_padding_z_with_zero(self, file):
        df = pd.read_csv(file)
        data = np.zeros((3, df.shape[0], self.max_joint, self.max_body))
        for n, row in df.iterrows():
            for m in range(self.max_body):
                for j in range(1, 35, 2):
                    x = row[j] / self.image_w
                    y = row[j + 1] / self.image_h
                    z = 1.0
                    if m == 0:
                        data[:, n, int((j - 1) / 2), m] = [x, y, z]
                    if j == 33:
                        break
        return data

    def read_kpss(self, file):
        df = pd.read_csv(file)
        # print(df.head())
        kpss = []
        for i, r in df.iterrows():
            kps = []
            for j in range(1, 35, 2):
                # print(j)
                kps.append([r[j], r[j + 1]])
                if j == 33:
                    break
            kpss.append(kps)
        return kpss

    def load_bone_data(self):
        process_bone()


if __name__ == '__main__':
    dir = "left"
    p = Process(rf"F:\pingpong-all-data\2023-9-5_总成数据集\7-7-keypoints\{dir}",
                f"7-7-{dir}", csv_name=f"pose-data-{dir}.csv")
    p.load_joint_data()
    p.load_bone_data()
