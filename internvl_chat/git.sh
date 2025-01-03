#!/bin/bash

# 提示用户输入 commit 信息
echo "请输入 commit 信息:"
read comments

# 检查是否输入了 commit 信息
if [ -z "$comments" ]; then
    echo "错误: commit 信息不能为空！"
    exit 1
fi

# 执行 git add .
git add .

# 执行 git commit -m
git commit -m "$comments"

# 执行 git push
git push

# 提示完成
# echo "代码已成功提交并推送！"