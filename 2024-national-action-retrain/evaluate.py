"""
@Project: 2023-pp-action-plugin
@FileName: evaluate.py
@Description: 评估动作分类效果
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/2/2 23:44 at PyCharm
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="./simsun.ttc")


def plot_confusion_matrix(cm, savename, title='Confusion Matrix', classes=[]):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90, fontproperties=font)
    plt.yticks(xlocations, classes, fontproperties=font)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png', dpi=400)
    plt.show()


# classes = ['A', 'B', 'C', 'D', 'E', 'F']
#
# random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
# y_true = random_numbers.copy()  # 样本实际标签
# random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
# y_pred = random_numbers  # 样本预测标签
#
# # 获取混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

def test():
    classes = ["a", "b", "c"]
    y_true = ["c", "b", "c", "c"]
    y_pred = ["b", "a", "a", "c"]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')
    print(confusion_matrix(y_true, y_pred))

def eval(): # 2024/2/2 ASE-GCN 球星挑战赛 307 samples, 61 test reults
    y_pred = [3, 3, 9, 1, 10, 7, 1, 1, 3, 4, 3, 12, 1, 12, 1, 4, 4, 3, 1, 1, 3, 3, 1, 12, 9, 9, 12, 10, 1, 3, 4, 1, 4, 1, 10, 9, 10, 12, 3, 3, 1, 1, 3, 3, 1, 9, 12, 9, 12, 3, 1, 1, 9, 1, 1, 4, 9, 1, 12, 4, 10]
    y_true = [3, 6, 10, 9, 10, 7, 1, 1, 3, 4, 3, 12, 1, 0, 1, 4, 4, 3, 9, 1, 3, 7, 1, 12, 8, 3, 12, 10, 1, 3, 4, 8, 9, 1, 10, 9, 10, 1, 3, 3, 12, 9, 6, 3, 1, 9, 12, 9, 1, 3, 9, 1, 1, 1, 1, 4, 9, 1, 1, 4, 10]
    labels = range(0, 14)
    print(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    pp_star_challenge = {
        0: "反手-拧",
        1: "反手-撕",
        2: "正手-快带",
        3: "正手-搓",
        4: "发球-顺侧旋",
        5: "反手-拨",
        6: "反手-弹",
        7: "反手-搓",
        8: "正手-被动",
        9: "正手-拉",
        10: "发球-勾手",
        11: "正手-挑",
        12: "反手-拉",
        13: "反手-被动"
    }

    classes = [pp_star_challenge[i] for i in labels]
    plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix', classes=classes)

if __name__ == '__main__':
    eval()