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

KEYPOINT = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
    17: 'right_index'
}


SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16], [5, 6]
]

COLORS = [[255, 1, 1], [255, 85, 1], [255, 170, 1], [255, 255, 1], [170, 255, 1], [85, 255, 1], [1, 255, 1],
          [1, 255, 85], [1, 255, 170], [1, 255, 255], [220, 220, 220], [220, 220, 220], [220, 220, 220], [85, 1, 255],
          [170, 1, 255], [255, 1, 255], [255, 1, 170], [220, 220, 220]]


def draw_skeleton_kps_on_back_save_fig(kps, filename, mask=[1, 2], size=(1920, 1080)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for i in range(len(SKELETON)):
        if SKELETON[i][0] in mask:
            continue
        a_x, a_y = kps[SKELETON[i][0]][0], kps[SKELETON[i][0]][1]
        b_x, b_y = kps[SKELETON[i][1]][0], kps[SKELETON[i][1]][1]
        cv2.circle(img, (int(a_x), int(a_y)), 6, COLORS[i], -1)
        cv2.circle(img, (int(b_x), int(b_y)), 6, COLORS[i], -1)
        cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), COLORS[i], 2)
        cv2.imencode('.jpg', img)[1].tofile(filename)


def draw_skeleton_kps_on_origin(kps, img, mask=[1, 2], ratio=1):
    for i in range(len(SKELETON)):
        if SKELETON[i][0] in mask:
            continue
        a_x, a_y = kps[SKELETON[i][0]][0], kps[SKELETON[i][0]][1]
        b_x, b_y = kps[SKELETON[i][1]][0], kps[SKELETON[i][1]][1]
        # cv2.circle(img, (int(a_x), int(a_y)), 6, COLORS[i], -1)
        # cv2.circle(img, (int(b_x), int(b_y)), 6, COLORS[i], -1)
        # cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), COLORS[i], 2)
        # 同一绘图颜色
        color = (255, 255, 0)
        radius = int(10 * ratio)
        line_thickness = math.ceil(3 * ratio)
        cv2.circle(img, (int(a_x), int(a_y)), radius, color, -1)
        cv2.circle(img, (int(b_x), int(b_y)), radius, color, -1)
        cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), color, thickness=line_thickness)
    return img


def draw_box(img, p1, p2, ratio=1):
    cv2.rectangle(img, p1, p2, color=(255, 255, 255), thickness=math.ceil(3*ratio))
    return img

def draw_text(img, p1, text=""):
    org = p1
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    fontcolor = (255, 255, 255)  # BGR
    thickness = 1
    lineType = 4
    bottomLeftOrigin = 1
    # cv.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType, bottomLeftOrigin)
    cv2.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType)
    return img

def results_visualization(file_path):
    with open(file_path, 'rb') as file:
        data = json.load(file)

    # Extract 'predict_labels' from the data
    all_predict_labels = [item['predict_labels'][0] for item in data.values() if item['predict_labels']]

    # Count the frequency of each label
    label_counts = Counter(all_predict_labels)

    # Preparing data for plotting
    labels, counts = zip(*label_counts.items())

    # Setting matplotlib to display Chinese characters
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Or your preferred font
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Plotting the distribution
    plt.figure(figsize=(15, 8))
    plt.bar(labels, counts)
    plt.xlabel('预测标签')
    plt.ylabel('频率')
    plt.title('预测标签的分布')
    plt.xticks(rotation=90)
    plt.savefig("distribution.png", dpi=400)


def test():
    fig = plt.figure(num=1, figsize=(4, 4))
    plt.plot([1, 2, 3, 4], [1, 2, 3, 4])
    plt.show()

if __name__ == "__main__":
    # results_visualization(r"E:\pingpong-all-data\2024-1-25_国家队技术评估_动作分类\actions_results\2024_1_25_20_43_8_results.json")
    test()

