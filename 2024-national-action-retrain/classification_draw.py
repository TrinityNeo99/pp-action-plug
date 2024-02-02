'''
Descripttion: 骨骼关节点绘图工具
Author: Wei Jiangning
version: v 1.2
Date: 2022-12-03 17:27:57
LastEditors: Wei Jiangning
LastEditTime: 2022-12-05 23:31:20
'''
import math

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import json

from matplotlib.font_manager import FontProperties

def results_visualization(file_path):
    with open(file_path, 'rb') as file:
        data = json.load(file)

    # Extract 'predict_labels' from the data
    all_predict_labels = [item['predict_labels'][0] for item in data.values() if item['predict_labels']]

    # Count the frequency of each label
    label_counts = Counter(all_predict_labels)

    # Preparing data for plotting
    labels, counts = zip(*label_counts.items())

    
    font1 = FontProperties(fname="./simsun.ttc")

    # Plotting the distribution
    plt.figure(figsize=(15, 8))
    plt.bar(labels, counts)
    plt.xlabel('预测标签', fontproperties=font1)
    plt.ylabel('频率', fontproperties=font1)
    plt.title('预测标签的分布', fontproperties=font1)
    plt.xticks(rotation=45, fontproperties=font1)
    plt.savefig("distribution.png", dpi=400)
    plt.show()


def test():
    fig = plt.figure(num=1, figsize=(4, 4))
    plt.plot([1, 2, 3, 4], [1, 2, 3, 4])
    plt.show()

if __name__ == "__main__":
    results_visualization(r"./results/classification/2024_2_3_1_2_2_results.json")

