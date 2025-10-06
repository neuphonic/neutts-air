#!/bin/bash

# 查找并杀死 run_api_server.py 进程
PID=$(pgrep -f "run_api_server.py")

if [ -z "$PID" ]; then
    echo "API server is not running."
else
    echo "Stopping API server (PID: $PID)..."
    kill $PID
    echo "Stopped."
fi
