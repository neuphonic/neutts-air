#!/bin/bash

# 设置工作目录（可选，根据你的项目结构调整）
cd "$(dirname "$0")"

# 启动 Python 服务，输出重定向到日志文件，忽略挂断信号
nohup python3 run_api_server.py > api_server.log 2>&1 &

# 打印后台进程 PID
echo "API server started with PID $!"
