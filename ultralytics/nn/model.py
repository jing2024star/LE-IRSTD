import torch

# 加载模型
model_path = '/root/IRST_YOLO/ultralytics-main/yolov8n.pt'  # 替换为你的模型文件路径
model = torch.load(model_path)

# 如果模型是使用 `torch.nn.Module` 定义的类，确保你使用相同的类来加载模型
# model = YourModelClass()
# model.load_state_dict(torch.load(model_path))

# 打印模型结构
print(model)