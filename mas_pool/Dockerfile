# 使用官方 Python 镜像
FROM python:3.9-slim
# 如果加载不出来，尝试镜像源
# FROM dockerpull.org/python:3.9-slim

# 安装必要的包，执行代码时需要的包
# RUN pip install --no-cache-dir numpy
# RUN pip install --no-cache-dir numpy
# RUN pip install --no-cache-dir numpy
# RUN pip install --no-cache-dir numpy

# 设置工作目录
WORKDIR /app

# 复制代码文件到容器
COPY . /app

# 设置默认命令
CMD ["python", "execute_code.py"]
