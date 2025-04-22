# 项目流程说明

## 1. 项目结构优化
首先优化项目结构，让代码更易于管理和扩展。在当前目录下创建如下结构：

```
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
```

## 2. 拆分代码
将当前代码拆分成不同的文件，提高代码的可维护性。

### src/model.py
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)
```

### src/inference.py
```python
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
```

## 3. 创建依赖文件
在项目根目录创建 `requirements.txt`，列出项目所需的依赖库。

### requirements.txt
```
blinker==1.9.0
click==8.1.8
filelock==3.18.0
Flask==3.1.0
fsspec==2025.3.2
importlib_metadata==8.6.1
itsdangerous==2.2.0
Jinja2==3.1.6
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.2.1
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
sympy==1.13.1
torch==2.6.0
triton==3.2.0
typing_extensions==4.13.2
Werkzeug==3.1.3
zipp==3.21.0

```

## 4. 创建 HTTP 服务
使用 Flask 框架将模型封装成 HTTP 服务。

### src/app.py
```python
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
```

## 5. 创建 Dockerfile
在项目根目录创建 `Dockerfile`，将应用打包成 Docker 镜像。

```Dockerfile
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
```

## 6. 构建和运行 Docker 镜像
在项目根目录创建BuildDocker.sh 构建和运行 Docker 容器：

```
#!/bin/bash

# 设置错误时退出
set -e

echo "开始构建 Docker 镜像..."

# 构建 Docker 镜像
docker build -t mle2mlops:latest .

echo "Docker 镜像构建完成"

echo "启动 Docker 容器..."

# 检查并清理已有容器
if docker ps -a --filter "name=mle2mlops" | grep -q mle2mlops; then
    echo "发现已存在的 mle2mlops 容器，正在停止并移除..."
    docker stop mle2mlops >/dev/null 2>&1
    docker rm mle2mlops >/dev/null 2>&1
fi

# 运行 Docker 容器
# -d: 后台运行
# --name: 指定容器名称
# --restart: 自动重启策略
docker run -d \
  --name mle2mlops \
  --restart unless-stopped \
  -p 5000:5000 \
  mle2mlops:latest

echo "Docker 容器已启动，服务运行在 http://localhost:5000"
```

## 7. 测试服务
创建test.sh 使用 `curl` 测试服务：

```
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [4.0, 6.0]}'

```

## 8. 持续集成和持续部署（CI/CD）
可以使用 GitHub Actions、GitLab CI/CD 等工具实现自动化的代码测试、构建和部署。以下是一个简单的 GitHub Actions 示例：

###.github/workflows/main.yml
```yaml
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
      run: docker build -t mle2mlops.
    - name: Run tests (示例，可根据实际情况添加测试代码)
      run: echo "Running tests..."
    - name: Deploy to production (示例，可根据实际情况修改部署方式)
      run: echo "Deploying to production..."
```

