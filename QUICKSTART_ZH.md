# FireRedChat 快速入门指南

> ⚡ 5 分钟快速启动你的语音 AI Agent

---

## 📋 前置检查

确保你的系统满足以下要求：

- ✅ Docker & Docker Compose 已安装
- ✅ NVIDIA GPU (推荐，用于 ASR/TTS 加速)
- ✅ 至少 8GB 可用内存
- ✅ 至少 30GB 可用磁盘空间

---

## 🚀 快速启动 (5 步)

### 步骤 1: 克隆项目 (2 分钟)

```bash
git clone --recurse-submodules https://github.com/FireRedTeam/FireRedChat.git
cd FireRedChat
```

### 步骤 2: 启动基础服务 (1 分钟)

```bash
cd docker
docker-compose up -d
```

等待服务启动：
```bash
# 检查服务状态
docker-compose ps

# 应该看到：
# redis       running
# livekitrtc  running
# livekitplay running
```

### 步骤 3: 部署 LLM (1 分钟)

使用 Ollama (最简单):

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull qwen2.5-7b

# 后台运行
ollama serve &
```

### 步骤 4: 配置并启动 Agent (1 分钟)

```bash
cd agents/examples

# 复制环境配置
cp .env.dev .env

# 编辑配置文件 (可选，默认配置通常可用)
# vim fireredchat_worker.py

# 安装依赖 (首次需要)
cd ..
pip install -e .
cd fireredchat-plugins
pip install -e livekit-plugins-firered
pip install -e livekit-plugins-fireredchat-pvad
pip install -e livekit-plugins-fireredchat-turn-detector

# 启动 Agent
cd ../examples
python3 fireredchat_worker.py dev
```

### 步骤 5: 开始对话！

打开浏览器访问：**http://localhost:3000**

点击 "Join Room" 开始对话！🎉

---

## 🔧 可选：部署 ASR/TTS 服务

如果需要完整的语音功能（推荐）：

### 部署 ASR 服务

```bash
cd fireredasr-server/server

# 下载模型 (约 2GB)
mkdir -p models
git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L

# 构建并运行
docker build -t fireredasr-service .
docker run -d -p 8000:8000 --gpus '"cuda:0"' -v $(pwd)/models:/app/models fireredasr-service
```

### 部署 TTS 服务

```bash
cd fireredtts-server

# 下载模型 (注意：仅限非商业用途)
huggingface-cli download --resume-download --repo-type model \
  FireRedTeam/FireRedTTS-1S --revision fireredtts1s_4_chat \
  --local-dir ./tts_4_chat

# 运行服务
docker run -td --name ttsserver \
  -v "$(pwd)/tts_4_chat/pretrained_models:/workspace/models/redtts" \
  -p 8081:8081 \
  crpi-byegxpnnsibfy3j1.cn-shanghai.personal.cr.aliyuncs.com/fireredchat/fireredtts1-server:latest \
  bash /workspace/run.sh --llm --svc_config_path /workspace/svc.yaml --port 8081
```

---

## ✅ 验证安装

### 检查服务状态

```bash
# 所有服务应该运行中
docker ps

# 端口监听检查
netstat -tlnp | grep -E '3000|6379|7880|8000|8081'
```

### 测试 ASR 服务

```bash
curl -X POST http://localhost:8000/audio/transcriptions \
  -F file="@test.wav"
```

### 测试 TTS 服务

```bash
curl http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好", "voice": "f531"}' \
  --output test.mp3
```

---

## 🎯 下一步

- 📖 阅读完整文档：`PROJECT_DOCUMENTATION.md`
- 🔧 自定义角色：编辑 `fireredchat_worker.py` 中的 `character` 字典
- 🧪 测试示例：查看 `agents/examples/` 目录
- 📝 查看 API 文档：`PROJECT_DOCUMENTATION.md#api-参考`

---

## ❓ 常见问题

### Q: WebUI 无法访问？
```bash
# 检查容器
docker ps | grep livekitplay

# 查看日志
docker logs livekitplay
```

### Q: Agent 无法连接？
```bash
# 确认 LiveKit 运行
docker logs livekitrtc

# 检查 Redis
docker logs redis
```

### Q: 语音识别失败？
- 确认 ASR 服务运行：`curl http://localhost:8000/health`
- 检查模型文件是否完整
- 确认 GPU 可用：`nvidia-smi`

---

## 📞 获取帮助

- 📚 完整文档：`PROJECT_DOCUMENTATION.md`
- 💬 Discord 社区：https://discord.gg/livekit
- 🐛 提交 Issue：https://github.com/FireRedTeam/FireRedChat/issues

---

_祝你使用愉快！🎉_
