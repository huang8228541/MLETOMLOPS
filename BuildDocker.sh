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