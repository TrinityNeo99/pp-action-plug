"""
@Project: 2023-pp-action-plugin
@FileName: main.py
@Description: 主函数
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/9/22 19:47 at PyCharm
"""
import os.path
import shutil

from public.config import *
from yolo_keypoints import YoloExtractor
from keypointConvert import Process
from angular_skeleton_encoding.main import ase_gcn_pp_classify_api


class Action:
    def __init__(self, out_dir="temp"):
        self.yolo = YoloExtractor("", out_dir=out_dir, isBatchProcess=False)
        self.out_dir = out_dir

    def after_process(self, video_raw_name):
        shutil.rmtree(os.path.join(self.out_dir, video_raw_name))
        shutil.rmtree("./work_dir")
        pass

    def video2keypoint(self, video_path):
        args = action_config(video_path=video_path, keypoints_dir=self.out_dir)
        video_raw_name = os.path.basename(video_path).split(".")[0]
        self.yolo.extract(args, save_key_points_csv=True)
        return video_raw_name

    def convert2npy(self, video_raw_name, person_pose):
        data_dir = self.out_dir
        out_dir = os.path.join(self.out_dir, video_raw_name)
        convertor = Process(data_path=data_dir, out_path=out_dir, csv_name=f"pose-data-{person_pose}.csv")
        convertor.load_joint_data()
        # convertor.load_bone_data()

    def run(self, video_path, person_pose: str, save_delete_middle_files=True):
        '''
        for multiple video usage, just run this function for multi times
        :param video_path: 视频路径
        :param person_pose: 人员位置: "left" / "right"
        :return: top3 labels
        '''

        video_raw_name = self.video2keypoint(video_path)
        self.convert2npy(video_raw_name, person_pose)
        predict_labels, predict_labels_top3 = ase_gcn_pp_classify_api(video_raw_name)
        if save_delete_middle_files:
            self.after_process(video_raw_name)
        return predict_labels, predict_labels_top3


if __name__ == '__main__':
    a = Action()
    predict_labels, predict_labels_top3 = a.run(
        r"F:\pingpong-all-data\2023-9-5_总成数据集\6-6-split\2023WTT新乡第二局6-6_Sub_01.mp4", "left")
    print("this is the a.run out", predict_labels, predict_labels_top3)
