

1. 项目结构优化
首先优化项目结构，让代码更易于管理和扩展。在当前目录下创建如下结构：


mle2mlops/
├── data/                # 存放数据集
├── models/              # 存放模型权重文件
│ └── model.pth
├── src/
│ ├── model.py         # 模型定义
│ └── inference.py     # 推理代码
├── Dockerfile           # Docker 配置文件
├── requirements.txt     # 依赖文件
└── MLE_handover.ipynb   # 原始 Notebook
2. 拆分代码
将当前代码拆分成不同的文件，提高代码的可维护性。

src/model.py

import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)
src/inference.py

import torch
from src.model import SimpleModel

# 加载模型
model = SimpleModel()
model.load_state_dict(torch.load("../models/model.pth"))
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


3. 创建依赖文件
在项目根目录创建 requirements.txt，列出项目所需的依赖库。


requirements.txt
torch


4. 创建 HTTP 服务
使用 Flask 框架将模型封装成 HTTP 服务。

src/app.py

from flask import Flask, request, jsonify
from inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    input_data = data.get('input', [])
    result = predict(input_data)
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


5. 创建 Dockerfile
在项目根目录创建 Dockerfile，将应用打包成 Docker 镜像。


# 使用 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "src/app.py"]



6. 构建和运行 Docker 镜像
在项目根目录下执行以下命令构建和运行 Docker 容器：


# 构建 Docker 镜像
docker build -t mle2mlops .

# 运行 Docker 容器
docker run -p 5000:5000 mle2mlops


7. 测试服务
使用 curl 或者 Postman 测试服务：

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"input": [4.0, 6.0]}'

8. 持续集成和持续部署（CI/CD）
可以使用 GitHub Actions、GitLab CI/CD 等工具实现自动化的代码测试、构建和部署。以下是一个简单的 GitHub Actions 示例：

.github/workflows/main.yml

name: CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: docker build -t mle2mlops .
    - name: Run tests (示例，可根据实际情况添加测试代码)
      run: echo "Running tests..."
    - name: Deploy to production (示例，可根据实际情况修改部署方式)
      run: echo "Deploying to production..."

通过以上步骤，你就可以将机器学习模型从开发环境部署到生产环境，并实现基本的 MLOps 流程。
