import torch
from model import SimpleModel

# 加载模型
model = SimpleModel()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

def predict(input_data):
    """
    进行模型推理
    :param input_data: 输入数据，格式为列表 [x1, x2]
    :return: 预测结果
    """
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()