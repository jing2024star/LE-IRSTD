from ultralytics.yolo.utils.configfile import __predict
from ultralytics import YOLO

def predictmodel(predictConfig,sourceConfig):
      # 开始加载模型
      model = YOLO(__predict(predictConfig))
      # 指定训练参数开始测试
      for i in model.predict(source=sourceConfig, stream=True, conf=0.6, iou=0.5,
                             project="/root/IRST_YOLO/ultralytics-main/runs/predict", name='IRSTD-1', save_txt=True, save=True, show_conf=True):
            print(i)

if __name__ == "__main__":
      # 填写测试的网络模型名称
      predictConfig = "/root/IRST_YOLO/ultralytics-main/runs/train/IRSTD1/weights/best.pt"

      # 填写测试图片文件夹
      sourceConfig = '/root/IRST_YOLO/ultralytics-main/datasets/IRSTD/images/test'

      # 调用测试方法
      predictmodel(predictConfig,sourceConfig)

























































