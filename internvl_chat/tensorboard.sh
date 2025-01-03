#!/bin/bash

# 检查输入参数
if [ $# -lt 1 ]; then
  echo "Usage: $0 <logdir> [port] [host]"
  exit 1
fi

# 获取路径
LOGDIR=$1

# 设置默认端口和主机
PORT=${2:-10787}
HOST=${3:-0.0.0.0}

# 启动 TensorBoard
echo "Starting TensorBoard with logdir=$LOGDIR, port=$PORT, host=$HOST"
tensorboard --logdir "$LOGDIR" --port "$PORT" --host "$HOST"