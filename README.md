# MLOps 需将模型封装为 HTTP 服务，使其可在 Docker 环境中运行，保障服务 24x7 稳定运行，并支持未来模型升级

## 1. 代码结构优化
在当前目录下创建如下结构：

```
mletomlops/

.github/
└── workflows/
    └── main.yml  

项目文件：
├── BuildDocker.sh
├── Dockerfile
├── MLE_handover.ipynb
├── README.md
├── models/
│   └── model.pth
├── requirements.txt
├── src/
│   ├── app.py
│   ├── inference.py
│   └── model.py
└── test.sh

```

## 2. 拆分代码
将代码拆分成不同的文件。

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
在项目根目录创建 `requirements.txt`。

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
在根目录创建BuildDocker.sh 构建和运行 Docker 容器：

```sh
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
创建test.sh 使用 `curl` 测试：

```sh
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [4.0, 6.0]}'

```

## 8. 持续集成和持续部署（CI/CD）

###.github/workflows/main.yml
```yml
name: CI/CD

on:
  push:
    branches: [ main ]
  

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      # 检出代码到运行环境
      - name: Checkout code
        uses: actions/checkout@v4

      # 设置 Python 环境
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9

      # 安装项目依赖
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # 构建 Docker 镜像
      - name: Build Docker image
        run: docker build -t mle2mlops .

      # 登录 Docker Hub
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      # 标记并推送 Docker 镜像到 Docker Hub
      - name: Tag and push Docker image
        run: |
          docker tag mle2mlops ${{ secrets.DOCKERHUB_USERNAME }}/mle2mlops:latest
          docker push ${{ secrets.DOCKERHUB_USERNAME }}/mle2mlops:latest

      # 模拟部署到生产环境，SSH 连接到服务器并拉取新镜像
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USERNAME }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            docker pull ${{ secrets.DOCKERHUB_USERNAME }}/mle2mlops:latest
            docker stop mle2mlops || true
            docker run -d -p 5000:5000 --name mle2mlops ${{ secrets.DOCKERHUB_USERNAME }}/mle2mlops:latest
```

