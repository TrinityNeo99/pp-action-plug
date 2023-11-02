"""
@Project: 2023-yolo-fun
@FileName: yolo_keypoints.py
@Description: yolo 处理视频形成关键点
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/9/7 16:56 at PyCharm
"""
import os.path

from ultralytics import YOLO
from public.utils import *
from public.draw import *
from public.config import *
import cv2
import csv

COCO_KEYPOINT_INDEXES = {
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
    16: 'right_ankle'
}


class YoloExtractor():

    def __init__(self, video_dir, out_dir="./profile_output", isBatchProcess=True):
        self.model = YOLO('./resource/yolov8m-pose.pt')  # load a pretrained model (recommended for training)
        if isBatchProcess:
            self.data_dir = video_dir
            self.out_dir = out_dir
            self.files = os.listdir(self.data_dir)

    def yolo_keypoints_on_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(image, device="0", save_conf=True, conf=0.25, verbose=False)
        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints

            # select the bottom most two boxes
            box_id2right_y = [(b, boxes[b].xyxy[0][3]) for b in range(len(boxes))]
            sorted_box_id2right_y = sorted(box_id2right_y, key=lambda item: item[1], reverse=True)
            box_id_top2 = [sorted_box_id2right_y[0][0], sorted_box_id2right_y[1][0]]

            for id in box_id_top2:
                kps = keypoints[id].xy.cpu().numpy().tolist()[0]
                bounding = boxes[id].xyxy.cpu().numpy().tolist()[0]
                # draw_skeleton_kps_on_origin(kps, image)  # the shape of kps is (1,17,2)
                draw_box(image, p1=(int(bounding[0]), int(bounding[1])), p2=(int(bounding[2]), int(bounding[3])))
                # draw_text(image, p1=(int(bounding[0]), int(bounding[1] - 20)), text="player")

            # the kps of left person and right person
            if boxes[box_id_top2[0]].xyxy[0][0] < boxes[box_id_top2[1]].xyxy[0][0]:
                id_left, id_right = box_id_top2[0], box_id_top2[1]
            else:
                id_left, id_right = box_id_top2[1], box_id_top2[0]

            left_kps = keypoints[id_left].xy.cpu().numpy().tolist()[0]
            right_kps = keypoints[id_right].xy.cpu().numpy().tolist()[0]

            # flat two-dim list 2 one-dim list
            left_kps = two_list_flat(left_kps)
            right_kps = two_list_flat(right_kps)
        return image, left_kps, right_kps

    def extract(self, args, save_key_points_csv=False, save_infer_video=False):
        frame_num, fps, duration, width, height = video_info(args.video_path)
        keypoints_video_out_path = os.path.join(args.keypoints_dir, args.video_infer_raw_name)
        vout = get_vout_H264_mp4(keypoints_video_out_path)
        cap = cv2.VideoCapture()
        cap.open(args.video_path)
        l_kpss, r_kpss = [], []
        cnt = 0
        with tqdm(total=frame_num) as pbar:
            while cap.isOpened():
                success, image = cap.read()
                cnt += 1
                pbar.update(1)
                if not success:
                    break
                image_ske, l_kps, r_kps = self.yolo_keypoints_on_image(image)
                if save_infer_video:
                    vout.append_data(image_ske)
                if save_key_points_csv:
                    l_kps.insert(0, cnt)
                    r_kps.insert(0, cnt)
                    l_kpss.append(l_kps)
                    r_kpss.append(r_kps)
        if save_key_points_csv:
            self.wirte_csv(args, l_kpss, out_name="pose-data-left.csv")
            self.wirte_csv(args, r_kpss, out_name="pose-data-right.csv")
        return l_kpss, r_kpss


    def wirte_csv(self, args, csv_output_rows, out_name="pose-data.csv"):
        csv_headers = ['frame']
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint + '_x', keypoint + '_y'])

        csv_output_filename = os.path.join(args.keypoints_dir, out_name)
        print(csv_output_filename)
        with open(csv_output_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_headers)
            csvwriter.writerows(csv_output_rows)

    def run(self):
        start = time.time()
        for f in self.files:
            logger.info(f"processing {f} ({self.files.index(f) + 1}/{len(self.files)})")
            args = action_config(video_path=os.path.join(self.data_dir, f), keypoints_dir=self.out_dir)
            l_kpss, r_kpss = self.extract(args, save_key_points_csv=False)
        end = time.time()
        logger.info(f"Total cost: {round((end - start), 2)} second")


if __name__ == '__main__':
    y = YoloExtractor(r"F:\pingpong-all-data\2023-9-5_总成数据集\source2", out_dir=r"./temp")
    y.run()
