###
## Make both ends meet: A synergistic optimization infrared small target detection with streamlined computational overhead [[PDF](https://doi.org/10.1016/j.optlastec.2026.114661)]


## Environment

```
pip install -r requirements.txt
```

## 1.数据集准备
datasets/[数据集名称]/
        ├── images/
        │   ├── train/
        │   ├── val/datasets
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

## 2.请确保 datasets.yaml 配置正确：
path: /root/IRST_YOLO/ultralytics-main/datasets/IRSTD # dataset root dir
train: images/train 
val: images/val  
test: images/test  

# Classes
names:
  0: target
  ...

## How To Train

· ultralytics/models/v8/yolov8_4_Faststem_AVSFPN.yaml 改 nc: [类别数量]
· python train.py

## How To Test

· python val.py
