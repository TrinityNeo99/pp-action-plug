"""
@Project: 2023-pp-action-plugin
@FileName: annotation.py
@Description: 处理人工标注的数据，与切分好的视频对应
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/1/30 22:24 at PyCharm
"""
import sys

import pandas as pd
import time
import json
import os
sys.path.append("./")
from main import Action, parse_filename
from tqdm import tqdm


def dict2json_format(d):
    return json.dumps(d, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)


def custom_game_mapping():
    map = {
        "1": "0-0",
        "2": "0-1",
        "3": "1-1",
        "4": "2-1",
        "5": "3-1",
    }
    return map


class Process:
    def __init__(self, anotation_path, clip_dir, results_path):
        self.annotation_results = None
        self.a_path = anotation_path
        self.c_dir = clip_dir
        self.results_path = results_path
        if "球星挑战赛" in anotation_path:
            self.map = custom_game_mapping()

    def parse_clip(self):
        self.clip_info = os.listdir(self.c_dir)
        if False:  # 用于检查标注数据是否错误
            for i in range(1, len(self.clip_info)):
                if self.clip_info[i][0:7] == self.clip_info[i - 1][0:7] and \
                        self.clip_info[i][-5] == self.clip_info[i - 1][-5] and \
                        int(self.clip_info[i].split("_")[2]) - int(self.clip_info[i - 1].split("_")[2]) == 1:
                    print(self.clip_info[i - 1])
                    print(self.clip_info[i])
                    print("")

    def parse_annotation(self, not_save=False):
        prune_data = pd.read_excel(self.a_path, sheet_name='prune')
        prune_data_cleaned = prune_data.dropna(how='all')
        results = {}
        game_order, score, order, action = '', '', '', ''
        for index, row in prune_data_cleaned.iterrows():
            if row["info"] == "局次":
                game_order = self.map[str(row['A'])]
            elif pd.isna(row["info"]):
                score = f"{row['B']}-{row['A']}"
            elif row["info"] == "板数" or row["info"] == "当前比分":
                pass
            else:
                order = row['info']
                direction = "L" if pd.isna(row['A']) == False else f"R"
                action = row['A'] if pd.isna(row['A']) == False else row['B']
                results[f"{game_order}_{score}_{order}_{direction}.mp4"] = action

        if not not_save:
            j = dict2json_format(results)
            local_time = time.localtime(time.time())
            timestamp = f"{local_time.tm_year}_{local_time.tm_mon}_{local_time.tm_mday}_{local_time.tm_hour}_{local_time.tm_min}_{local_time.tm_sec}"
            with open(os.path.join(self.results_path, f"{timestamp}_results.json"), "w+", encoding="utf-8") as fp:
                fp.write(j)

        self.annotation_results = results

    def check(self):
        self.parse_clip()
        self.parse_annotation(not_save=True)
        bad_samples = []
        for i in self.annotation_results.keys():
            if i not in self.clip_info:
                # check the bad samples
                # i = i.replace("R", "L") if "R" in i else i.replace("L", "R")
                # if i not in self.clip_info:
                bad_samples.append(i)
        self.bad_samples = bad_samples
        with open(os.path.join(self.results_path, "unmatched_samples.txt"), "w+") as fp:
            for s in bad_samples:
                fp.write(s + "\n")
        # print(bad_samples)
        # print(len(bad_samples))

    def convert_clips_skeletons(self):
        self.check()
        a = Action(self.results_path)
        with tqdm(total=len(self.annotation_results.keys()), desc="convert processing") as pbar:
            for c in self.annotation_results.keys():
                if c in self.bad_samples:
                    continue
                # direction = parse_filename(c)
                a.video2keypoint(os.path.join(self.c_dir, c))
                pbar.update(1)



if __name__ == '__main__':
    p = Process(anotation_path=r"E:\pingpong-all-data\2024-1-25_国家队技术评估_动作分类\球星挑战赛\annotation.xlsx",
                clip_dir=r"E:\pingpong-all-data\2024-1-25_国家队技术评估_动作分类\球星挑战赛\clips",
                results_path=r"E:\pingpong-all-data\2024-1-25_国家队技术评估_动作分类\球星挑战赛\clips_out")
    p.convert_clips_skeletons()
