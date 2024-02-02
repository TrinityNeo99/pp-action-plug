# pp-action-plug

## Usage
`python run main.py`


## pipeline
video, left -> yolo_get_keypoint -> csv -> feature_convert -> npy -> infer -> acton class

## 数据格式切换
pingpong-109-coco 29 classes, 在ase-gcn里用pp-lables

pingpong-star-challenge 14 classes 在ase-gcn里用pp-star-challenge

修改模型后，需要修改 win-test.yaml 里的模型权重 + 类别数