# FireRedChat 部署指南

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10  
> 🎯 版本：v1.0

---

## 📑 目录

1. [系统要求](#系统要求)
2. [快速部署](#快速部署)
3. [详细部署步骤](#详细部署步骤)
4. [配置说明](#配置说明)
5. [服务管理](#服务管理)
6. [故障排查](#故障排查)
7. [性能优化](#性能优化)
8. [生产环境部署](#生产环境部署)

---

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 | 生产环境 |
|------|----------|----------|----------|
| **CPU** | 4 核 | 8 核 | 16 核+ |
| **内存** | 8GB | 16GB | 32GB+ |
| **GPU** | 可选 | NVIDIA 8GB+ | NVIDIA 16GB+ |
| **存储** | 50GB | 100GB | 200GB+ SSD |
| **网络** | 100Mbps | 1Gbps | 1Gbps+ |

### GPU 要求

| GPU 型号 | CUDA 版本 | 适用场景 |
|----------|-----------|----------|
| RTX 3060 (12GB) | 11.8+ | 开发/测试 |
| RTX 3080 (10GB) | 11.8+ | 中小型部署 |
| RTX 4090 (24GB) | 12.4+ | 大型部署 |
| Tesla T4 (16GB) | 11.8+ | 云服务器 |
| A100 (40GB) | 12.4+ | 企业级部署 |

### 软件要求

```
操作系统：Ubuntu 20.04+ / macOS 12+ / Windows 11
Docker: 20.10+
Docker Compose: 2.0+
NVIDIA Driver: 520+ (GPU 部署)
CUDA: 11.8 或 12.4
Python: 3.10+
Node.js: 18+ (WebUI)
```

---

## 快速部署

### 一键部署脚本

```bash
# 1. 克隆项目
git clone --recurse-submodules https://github.com/FireRedTeam/FireRedChat.git
cd FireRedChat

# 2. 启动基础服务
cd docker
docker-compose up -d

# 3. 下载模型
cd ../fireredasr-server/server
mkdir -p models
git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L

# 4. 构建 ASR 服务
docker build -t fireredasr-service .

# 5. 运行 ASR 服务
docker run -d \
  -p 8000:8000 \
  --gpus '"cuda:0"' \
  -v $(pwd)/models:/app/models \
  fireredasr-service

# 6. 运行 TTS 服务
cd ../../fireredtts-server
huggingface-cli download --resume-download --repo-type model \
  FireRedTeam/FireRedTTS-1S --revision fireredtts1s_4_chat \
  --local-dir ./tts_4_chat

docker run -td --name ttsserver \
  --security-opt seccomp:unconfined \
  -v "$(pwd)/tts_4_chat/pretrained_models:/workspace/models/redtts" \
  -p 8081:8081 \
  crpi-byegxpnnsibfy3j1.cn-shanghai.personal.cr.aliyuncs.com/fireredchat/fireredtts1-server:latest \
  bash /workspace/run.sh --llm --svc_config_path /workspace/svc.yaml --port 8081 --http_uri=/v1/audio/speech

# 7. 启动 Agent Worker
cd ../agents/examples
cp .env.dev .env
# 编辑 .env 配置服务地址
python3 fireredchat_worker.py dev
```

### 验证部署

```bash
# 检查服务状态
docker ps

# 应看到以下服务:
# - redis
# - livekitrtc
# - livekitplay
# - fireredasr-service
# - ttsserver

# 测试 Redis
docker exec -it redis redis-cli -a dev123456 ping
# 输出：PONG

# 测试 LiveKit
curl http://localhost:7880

# 测试 ASR
curl -X POST http://localhost:8000/audio/transcriptions \
  -F file="@test.wav"

# 测试 TTS
curl http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好","voice":"f531"}' \
  --output test.mp3

# 访问 WebUI
# 浏览器打开：http://localhost:3000
```

---

## 详细部署步骤

### 步骤 1: 克隆项目

```bash
# 克隆主仓库 (包含子模块)
git clone --recurse-submodules https://github.com/FireRedTeam/FireRedChat.git
cd FireRedChat

# 验证子模块
git submodule status
# 应显示 agents 和 agents-playground 子模块
```

### 步骤 2: 配置环境变量

**创建 `.env` 文件**:

```bash
# docker/.env
REDIS_PASSWORD=dev123456
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret
```

**编辑 `docker-compose.yaml`** (可选):

```yaml
# 修改 Redis 密码
environment:
  REDISCLI_AUTH: your_secure_password

# 修改端口映射
ports:
  - "7880:7880"  # LiveKit 信令
  - "8000:8000"  # ASR
  - "8081:8081"  # TTS
  - "3000:3000"  # WebUI
```

### 步骤 3: 启动基础服务

```bash
cd docker
docker-compose up -d

# 查看日志
docker-compose logs -f

# 检查健康状态
docker-compose ps
# 所有服务应为 "healthy" 或 "running"
```

### 步骤 4: 部署 ASR 服务

#### 4.1 下载模型

```bash
cd fireredasr-server/server

mkdir -p models

# 下载标点模型
git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
cd models/PUNC-BERT && git lfs pull && cd ../..

# 下载 BERT 基座
git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
cd models/PUNC-BERT/chinese-lert-base && git lfs pull && cd ../../..

# 下载 ASR 模型
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L
cd models/FireRedASR-AED-L && git lfs pull && cd ../..
```

**模型大小**:
- PUNC-BERT: ~1.2GB
- chinese-lert-base: ~400MB
- FireRedASR-AED-L: ~2.5GB
- **总计**: ~4.1GB

#### 4.2 构建 Docker 镜像

**编辑 `Dockerfile`** (根据 GPU 调整 CUDA 版本):

```dockerfile
# Ampere 及更新 (RTX 3080, 4090, A100)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Volta (RTX 3080, V100)
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
```

**构建**:

```bash
# 中国大陆使用镜像加速
RUN pip install --no-cache-dir -r requirements.txt \
  --index-url http://mirrors.tencentyun.com/pypi/simple/ \
  --trusted-host mirrors.tencentyun.com

# 构建
docker build -t fireredasr-service .
```

#### 4.3 运行服务

```bash
# GPU 模式
docker run -d \
  -p 8000:8000 \
  --gpus '"cuda:0"' \
  -v $(pwd)/models:/app/models \
  -e FIREREDASR_PATH=/app/fireredasr \
  -e MODEL_DIR=/app/models \
  fireredasr-service

# CPU 模式 (无 GPU)
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e FIREREDASR_PATH=/app/fireredasr \
  -e MODEL_DIR=/app/models \
  fireredasr-service
```

### 步骤 5: 部署 TTS 服务

#### 5.1 下载模型

⚠️ **注意**: FireRedTTS 模型权重仅限**非商业用途**使用

```bash
cd fireredtts-server

# 下载模型
huggingface-cli download --resume-download --repo-type model \
  FireRedTeam/FireRedTTS-1S --revision fireredtts1s_4_chat \
  --local-dir ./tts_4_chat

# 或使用 git
git clone -b fireredtts1s_4_chat \
  https://huggingface.co/FireRedTeam/FireRedTTS-1S \
  tts_4_chat
```

**模型大小**: ~3GB

#### 5.2 运行服务

```bash
docker run -td --name ttsserver \
  --security-opt seccomp:unconfined \
  -v "$(pwd)/tts_4_chat/pretrained_models:/workspace/models/redtts" \
  -p 8081:8081 \
  crpi-byegxpnnsibfy3j1.cn-shanghai.personal.cr.aliyuncs.com/fireredchat/fireredtts1-server:latest \
  bash /workspace/run.sh \
    --llm \
    --svc_config_path /workspace/svc.yaml \
    --port 8081 \
    --http_uri=/v1/audio/speech
```

### 步骤 6: 部署 LLM 服务

#### 选项 A: Ollama (轻量级，推荐)

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama run qwen2.5-7b

# 后台运行
ollama serve

# 验证
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5-7b","prompt":"Hello"}'
```

**推荐模型**:
- `qwen2.5-7b`: 平衡性能和速度
- `qwen2.5-14b`: 更高质量
- `llama3-8b`: 英文场景

#### 选项 B: vLLM (高性能)

```bash
# Docker 运行
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096

# 验证
curl http://localhost:8000/v1/models
```

### 步骤 7: 配置 Agent Worker

#### 7.1 安装依赖

```bash
cd agents

# 安装主框架
pip install -e .

# 安装 FireRed 插件
cd fireredchat-plugins
pip install -e livekit-plugins-firered
pip install -e livekit-plugins-fireredchat-pvad
pip install -e livekit-plugins-fireredchat-turn-detector
```

#### 7.2 配置环境

**编辑 `.env`**:

```bash
cd agents/examples
cp .env.dev .env
```

**编辑 `.env` 内容**:

```bash
# LiveKit 配置
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=dev123456

# ASR 配置
FIREREDASR_BASE_URL=http://localhost:8000

# TTS 配置
FIREREDTTS_BASE_URL=http://localhost:8081/v1
FIREREDTTS_VOICE=f531

# LLM 配置
OPENAI_BASE_URL=http://localhost:11434/v1  # Ollama
OPENAI_API_KEY=notneeded
OPENAI_MODEL=qwen2.5-7b
```

#### 7.3 下载模型文件

```bash
# 下载 pVAD 和 EOU 模型
python3 fireredchat_worker.py download-files
```

#### 7.4 启动 Worker

```bash
# 开发模式
python3 fireredchat_worker.py dev

# 生产模式
python3 fireredchat_worker.py start
```

### 步骤 8: 访问 WebUI

```bash
# 浏览器打开
http://localhost:3000

# 或使用 Docker 中的 WebUI
# 已在 docker-compose 中启动
```

---

## 配置说明

### LiveKit 配置

**文件**: `docker/livekit.yaml`

```yaml
# LiveKit 服务器配置
port: 7880                    # HTTP 信令端口
rtc:
  port_range_start: 50000     # UDP 端口范围开始
  port_range_end: 60000       # UDP 端口范围结束
  tcp_port: 7881              # TCP 媒体端口

# Redis 连接
redis:
  address: redis:6379         # Redis 地址
  password: dev123456         # 密码
  db: 0                       # 数据库

# 日志
logging:
  level: info
  json: true

# CORS (WebUI 访问)
cors:
  allowed_origins:
    - http://localhost:3000
    - https://your-domain.com
```

### Agent Worker 配置

**文件**: `agents/examples/fireredchat_worker.py`

```python
# 角色配置
character = {
    "youyou": """
    你是悠悠，一个有灵魂的大语言模型助手...
    """,
    "nana": """
    你是娜娜，一名塔罗占卜师...
    """,
}

# LLM 配置
used_llm = openai.LLM.with_ollama(
    model="qwen2.5-7b",
    base_url="http://localhost:11434/v1",
)

# ASR 配置
stt=firered.STT(
    model="FireRedASR-AED-1",
    base_url="http://localhost:8000",
)

# TTS 配置
tts=firered.TTS(
    model="fireredtts1.0",
    voice="f531",
    base_url="http://localhost:8081/v1",
)

# VAD 配置
vad=fireredchat_pvad.VAD.load(
    activation_threshold=0.5,
    min_speech_duration=0.16,
)

# 轮次检测配置
turn_detection=ChineseModel(
    unlikely_threshold=0.08,
)
```

### WebUI 配置

**文件**: `agents-playground/.env.local`

```bash
# LiveKit 连接
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880

# 功能开关
NEXT_PUBLIC_FEATURES_VOICE=true
NEXT_PUBLIC_FEATURES_CHAT=true
```

---

## 服务管理

### 启动/停止服务

```bash
# 启动所有服务
cd docker
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启服务
docker-compose restart

# 查看状态
docker-compose ps
```

### 查看日志

```bash
# 所有服务日志
docker-compose logs -f

# 单个服务日志
docker-compose logs -f livekitrtc
docker-compose logs -f fireredasr

# 实时日志
docker logs -f ttsserver
```

### 服务健康检查

```bash
# Redis 健康检查
docker exec redis redis-cli -a dev123456 ping

# LiveKit 健康检查
curl http://localhost:7880

# ASR 健康检查
curl http://localhost:8000/docs

# TTS 健康检查
curl http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"测试","voice":"f531"}' \
  --output test.mp3
```

### 资源监控

```bash
# Docker 资源使用
docker stats

# GPU 使用
nvidia-smi

# 内存使用
free -h

# 磁盘使用
df -h
```

---

## 故障排查

### 常见问题

#### 1. Agent Worker 无法连接 LiveKit

**症状**: `Connection refused` 错误

**解决方案**:
```bash
# 检查 LiveKit 状态
docker ps | grep livekitrtc

# 查看日志
docker logs livekitrtc

# 确认端口开放
netstat -tlnp | grep 7880

# 重启 LiveKit
docker-compose restart livekitrtc
```

#### 2. ASR 服务响应慢

**症状**: 语音识别延迟高 (>5s)

**解决方案**:
```bash
# 检查 GPU 可用
nvidia-smi

# 检查模型加载
docker logs fireredasr | grep "model"

# 确认 CUDA 版本
docker exec fireredasr nvcc --version

# 调整 CUDA 版本 (Dockerfile)
# FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
```

#### 3. TTS 服务返回 500 错误

**症状**: `Internal Server Error`

**解决方案**:
```bash
# 检查 TTS 容器日志
docker logs ttsserver

# 确认模型路径
ls -la tts_4_chat/pretrained_models/

# 重启服务
docker restart ttsserver

# 重新下载模型
huggingface-cli download --resume-download \
  FireRedTeam/FireRedTTS-1S --revision fireredtts1s_4_chat
```

#### 4. WebUI 无法访问

**症状**: 浏览器显示 `Connection refused`

**解决方案**:
```bash
# 检查容器状态
docker ps | grep livekitplay

# 查看日志
docker logs livekitplay

# 确认 .env.local 配置
cat agents-playground/.env.local

# 重新构建
docker-compose up -d --build livekitplay
```

#### 5. 语音识别不准确

**症状**: 识别错误率高

**解决方案**:
```bash
# 检查音频质量 (建议 16kHz, 16bit)
ffprobe -i test.wav

# 确认 PUNC-BERT 模型加载
docker logs fireredasr | grep "PUNC"

# 调整 VAD 阈值
# 编辑 fireredchat_worker.py
vad=fireredchat_pvad.VAD.load(activation_threshold=0.3)  # 降低阈值
```

#### 6. GPU 内存不足

**症状**: `CUDA out of memory`

**解决方案**:
```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 限制模型内存
# 编辑 ASR 服务配置
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 使用更小的模型
# ASR: FireRedASR-AED-S (小模型)
# LLM: qwen2.5-3b
```

### 日志位置

```bash
# Docker 日志
/var/lib/docker/containers/<container-id>/<container-id>-json.log

# Agent Worker 日志
/workspace/logs/*.log

# 系统日志
journalctl -u docker
```

---

## 性能优化

### 1. GPU 优化

```bash
# 设置 GPU 性能模式
nvidia-smi -pm 1
nvidia-smi -ac 8100,1500  # 锁定频率

# 启用 Tensor Core
# 在代码中使用 FP16
model.half()
```

### 2. 批处理优化

```python
# ASR 批处理
stt = firered.STT(batch_size=4)

# TTS 批处理 (如有支持)
tts = firered.TTS(batch_size=2)
```

### 3. 缓存优化

```python
# 启用 LLM 缓存
llm = openai.LLM.with_ollama(
    model="qwen2.5-7b",
    cache_dir="/workspace/cache"
)

# Redis 缓存
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### 4. 网络优化

```yaml
# docker-compose.yaml
services:
  livekitrtc:
    network_mode: host  # 使用主机网络 (减少 NAT 开销)
```

### 5. 并发优化

```python
# Worker 并发数
WorkerOptions(
    max_jobs=10,          # 最大并发任务
    job_memory_warn_mb=1500,
)
```

---

## 生产环境部署

### 高可用配置

#### 1. Redis 集群

```yaml
# docker-compose.prod.yaml
services:
  redis:
    image: redis:6-alpine
    deploy:
      replicas: 3
    command: redis-server --cluster-enabled yes
```

#### 2. LiveKit 多节点

```yaml
# 多个 LiveKit 节点
services:
  livekitrtc-1:
    <<: *livekit-config
    ports:
      - "7880:7880"
  
  livekitrtc-2:
    <<: *livekit-config
    ports:
      - "7881:7880"  # 不同端口
```

#### 3. 负载均衡

```nginx
# nginx.conf
upstream livekit {
    server node1:7880;
    server node2:7880;
    server node3:7880;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://livekit;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 监控与告警

#### Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'livekit'
    static_configs:
      - targets: ['livekitrtc:7880']
  
  - job_name: 'asr'
    static_configs:
      - targets: ['fireredasr:8000']
  
  - job_name: 'tts'
    static_configs:
      - targets: ['ttsserver:8081']
```

#### Grafana 仪表盘

导入 Dashboard ID:
- LiveKit: `19342`
- Node Exporter: `1860`
- Docker: `17449`

### 备份策略

```bash
# Redis 备份
docker exec redis redis-cli -a dev123456 BGSAVE

# 模型备份
tar -czf models_backup.tar.gz models/

# 日志轮转
logrotate /etc/logrotate.d/docker
```

### 安全加固

```bash
# 修改默认密码
# docker/.env
REDIS_PASSWORD=your_secure_password
LIVEKIT_API_SECRET=your_secret_key

# 启用 HTTPS
# 配置 nginx SSL
# 使用 Let's Encrypt 证书

# 防火墙规则
ufw allow 7880/tcp   # LiveKit
ufw allow 8000/tcp   # ASR
ufw allow 8081/tcp   # TTS
ufw allow 3000/tcp   # WebUI
```

---

## 附录：快速参考

### 端口映射

| 服务 | 端口 | 协议 |
|------|------|------|
| LiveKit (信令) | 7880 | HTTP/WebSocket |
| LiveKit (媒体) | 7881 | TCP |
| LiveKit (媒体) | 50000-60000 | UDP |
| Redis | 6379 | TCP |
| ASR | 8000 | HTTP |
| TTS | 8081 | HTTP |
| WebUI | 3000 | HTTP |
| LLM (Ollama) | 11434 | HTTP |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `REDIS_PASSWORD` | dev123456 | Redis 密码 |
| `LIVEKIT_API_KEY` | devkey | LiveKit API 密钥 |
| `FIREREDASR_BASE_URL` | http://localhost:8000 | ASR 地址 |
| `FIREREDTTS_BASE_URL` | http://localhost:8081/v1 | TTS 地址 |
| `OPENAI_BASE_URL` | http://localhost:11434/v1 | LLM 地址 |

### 常用命令

```bash
# 重启所有服务
docker-compose down && docker-compose up -d

# 清理缓存
docker system prune -a

# 更新模型
cd models && git pull

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 压力测试
ab -n 1000 -c 10 http://localhost:8000/docs
```

---

_部署指南文档结束_
