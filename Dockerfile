# 1. 使用官方轻量级 Python 镜像
FROM python:3.12-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统级依赖 (Pillow 处理图片需要一些底层 C 库)


# 4. 复制依赖清单并安装
# 先复制 requirements.txt 是为了利用 Docker 的缓存机制，加速后续构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制项目代码
# 注意：.gitignore 中忽略的文件不会被复制进去
COPY . .

# 6. 创建必要的静态文件目录（确保容器内目录存在）
RUN mkdir -p app/static/uploads app/static/results logs

# 7. 暴露 FastAPI 默认端口
EXPOSE 8000

# 8. 启动命令
# 使用 uvicorn 运行 FastAPI，--host 0.0.0.0 确保外部可以访问
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

