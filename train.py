from ultralytics import YOLO
from ultralytics.yolo.utils.configfile import __train
import warnings
import argparse
warnings.filterwarnings('ignore')


'''

# 这是要填写的训练的网络模型名称                                                                                                                                  train-mAP50    
ultralytics/models/v8/yolov8n.yaml                       summary: 225 layers, 3011043 parameters, 3011027 gradients, 8.2 GFLOPs          72.7              
ultralytics/models/v8/yolov8n_2_Faststem.yaml            summary: 205 layers, 2925607 parameters, 2925591 gradients, 8.3 GFLOPs          75.5             
ultralytics/models/v8/yolov8n_3_AVSFPN.yaml              summary: 321 layers, 2663611 parameters, 2663595 gradients, 7.4 GFLOPs          74.5                                  
ultralytics/models/v8/yolov8n_4_Faststem_AVSFPN.yaml     summary: 301 layers, 2578175 parameters, 2578159 gradients, 7.5 GFLOPs          78.7
'''

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='', help='loading pretrain weights')
    parser.add_argument('--cfg', type=str, default='ultralytics/models/v8/yolov8_4_Faststem_AVSFPN.yaml', help='models')# 填写训练的网络模型名称
    parser.add_argument('--data', type=str, default='/root/IRST_YOLO/ultralytics-main/datasets.yaml', help='datasets')
    parser.add_argument('--epochs', type=int, default=50, help='train epoch')
    parser.add_argument('--batch', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=640, help='image sizes')
    parser.add_argument('--optimizer', default='Adam', help='use optimizer')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=32, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='IRSTD', help='save to project/name')
    parser.add_argument('--iou', type=float, default='0.7')
    parser.add_argument('--lr0', type=float, default='0.001')
    parser.add_argument('--close_mosaic', type=int, default='0')
    parser.add_argument('--resume', type=bool, default='False')
    parser.add_argument('--dropout', type=float, default='0.0')
    parser.add_argument('--cos_lr', type=bool, default='False')
    parser.add_argument('--save_conf', type=bool, default='True')
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain (scale with pixels)')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    return parser.parse_args()

if __name__ == '__main__':
    args=main()
    # 开始加载模型
    model = YOLO(__train(args.cfg))
    if(".pt" in args.weights):
        print("+++++++载入预训练权重：",args.weights,"++++++++")
        model.load(args.weights)
    else:
        print("-------没有载入预训练权重-------")
    # 指定训练参数开始训练
    model.train(data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                optimizer=args.optimizer,
                device=args.device,
                workers=args.workers,
                project=args.project,
                name=args.name,
                iou=args.iou,
                lr0=args.lr0,
                close_mosaic=args.close_mosaic,
                resume=args.resume,
                dropout=args.dropout,
                cos_lr=args.cos_lr,
                save_conf=args.save_conf,
                box=args.box,
                cls=args.cls,
                dfl=args.dfl)