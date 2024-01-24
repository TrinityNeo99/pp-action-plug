"""
@Project: 2023-pp-action-plugin
@FileName: main.py
@Description: 主函数
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/9/22 19:47 at PyCharm
"""
import json
import os.path
import shutil
import time

from public.config import *
from yolo_keypoints import YoloExtractor
from keypointConvert import Process
from angular_skeleton_encoding.main import ase_gcn_pp_classify_api


class Action:
    def __init__(self, out_dir="temp"):
        self.yolo = YoloExtractor("", out_dir=out_dir, isBatchProcess=False)
        self.out_dir = out_dir

    def after_process(self, video_raw_name):
        # time.sleep(1)
        # shutil.rmtree(os.path.join(self.out_dir, video_raw_name))
        shutil.rmtree("./work_dir")
        pass

    def video2keypoint(self, video_path):
        args = action_config(video_path=video_path, keypoints_dir=self.out_dir)
        video_raw_name = os.path.basename(video_path).split(".")[0]
        self.yolo.extract(args, save_key_points_csv=True, save_infer_video=True)
        return video_raw_name

    def convert2npy(self, video_raw_name, person_pose):
        data_dir = self.out_dir
        out_dir = os.path.join(self.out_dir, video_raw_name)
        convertor = Process(data_path=data_dir, out_path=out_dir, csv_name=f"pose-data-{person_pose}.csv")
        # convertor.load_joint_data()
        convertor.load_joint_data_single(video_raw_name)
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


def dict2json_format(d):
    return json.dumps(d, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)


def batch_run(dir_path, result_output_path, direction="left"):
    a = Action()
    fs = os.listdir(dir_path)
    results = {}
    for f in fs:
        predict_labels, predict_labels_top3 = a.run(os.path.join(dir_path, f), direction)
        results[f"{f}-{direction}"] = {"name": "a", "predict_labels": predict_labels, "predict_labels_top3": predict_labels_top3}
    j = dict2json_format(results)
    local_time = time.localtime(time.time())
    timestamp = f"{local_time.tm_year}_{local_time.tm_mon}_{local_time.tm_mday}"
    with open(os.path.join(result_output_path, f"{timestamp}_{direction}_results.json"), "w+", encoding="utf-8") as fp:
        fp.write(j)


def demo(path):
    a = Action()
    # path = r"./resource/2023WTT新乡第二局6-6_Sub_01.mp4"
    # path = r"E:\发球\serve\41.mp4"
    path = r"E:\pingpong-all-data\2023-11-2-发球测试数据集\serve\4.mp4"
    predict_labels, predict_labels_top3 = a.run(path, "left")
    print("this is the a.run out", predict_labels, predict_labels_top3)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some args.')
    parser.add_argument('--batch_process', "-b", default=False, help="batch_process or not",
                        action="store_true")
    parser.add_argument('--path', "-p", default="", required=False, type=str,
                        help="directory path for batch process or file path for demo")
    parser.add_argument('--out_path', "-o", default='./temp', type=str, required=False,
                        help="batch process output path")
    parser.add_argument('--direction', "-dr", default='left', type=str, required=False,
                        help="batch process output path")
    args = parser.parse_args()
    if args.batch_process == True:
        batch_run(args.path, args.out_path, args.direction)
    else:
        demo(args.path)
