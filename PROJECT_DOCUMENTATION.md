# FireRedChat 项目详细文档

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-09  
> 📦 项目版本：基于 FireRedChat 主分支

---

## 📑 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [核心组件详解](#核心组件详解)
4. [部署指南](#部署指南)
5. [API 参考](#api 参考)
6. [代码结构](#代码结构)
7. [配置说明](#配置说明)
8. [使用示例](#使用示例)
9. [故障排查](#故障排查)
10. [开发指南](#开发指南)

---

## 项目概述

### 什么是 FireRedChat？

FireRedChat 是一个**完全自托管的实时语音交互解决方案**，用于构建全双工语音 AI Agent。它集成了以下核心功能：

- 🔊 **TTS** (Text-to-Speech) - 文本转语音
- 🎤 **ASR** (Automatic Speech Recognition) - 自动语音识别
- 🎯 **pVAD** (Personalized Voice Activity Detection) - 个性化语音活动检测
- 🔄 **EoT** (End-of-Turn) - 轮次结束检测

### 核心优势

| 特性 | 说明 |
|------|------|
| 🚫 无外部 API 依赖 | 完全自托管，无需依赖第三方服务 |
| 🔒 零数据泄露 | 所有数据在本地处理，保障隐私安全 |
| 🎛️ 完全部署控制 | 可自定义配置和扩展 |

### 技术栈

- **核心框架**: LiveKit Agents (fork 版本)
- **RTC 服务器**: LiveKit RTC Server
- **Web 界面**: Next.js + TypeScript
- **后端服务**: Python (FastAPI, PyTorch)
- **消息队列**: Redis
- **AI 模型**: 
  - FireRedASR-AED-L (语音识别)
  - FireRedTTS-1S (语音合成)
  - PUNC-BERT (标点模型)
  - ChineseModel (中文轮次检测)

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户浏览器/客户端                        │
│                    (agents-playground WebUI)                 │
│                         Port: 3000                           │
└────────────────────┬────────────────────────────────────────┘
                     │ WebSocket (WSS)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    LiveKit RTC Server                        │
│                    Port: 7880 (HTTP) / 7881 (TCP)            │
│                    UDP: 50000-60000 (媒体流)                  │
└──────┬──────────────────────────────────────┬────────────────┘
       │                                      │
       ▼                                      ▼
┌──────────────┐                    ┌──────────────────┐
│   Redis      │                    │  AI Agent Worker │
│  Port: 6379  │                    │   (Python)       │
│  (状态存储)   │                    │                  │
└──────────────┘                    └────────┬─────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
           ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
           │   FireRedASR   │     │   FireRedTTS   │     │     LLM        │
           │   Port: 8000   │     │   Port: 8081   │     │   (Ollama等)    │
           │   (语音识别)    │     │   (语音合成)    │     │                │
           └────────────────┘     └────────────────┘     └────────────────┘
```

### 数据流

1. **用户语音输入** → LiveKit RTC → Agent Worker
2. **语音识别** → FireRedASR → 文本
3. **LLM 处理** → 生成回复文本
4. **语音合成** → FireRedTTS → 音频
5. **音频输出** → LiveKit RTC → 用户

---

## 核心组件详解

### 1. LiveKit RTC Server

**功能**: 实时音视频通信核心

**配置**:
- HTTP 端口：7880
- TCP 端口：7881
- UDP 端口范围：50000-60000 (媒体流)

**关键文件**: `docker/livekit.yaml`

```yaml
port: 7880
redis:
  address: 0.0.0.0:6379
  password: dev123456
  db: 0
rtc:
  port_range_start: 50000
  port_range_end: 60000
  tcp_port: 7881
```

### 2. Redis Server

**功能**: 多节点状态存储和消息队列

**配置**:
- 端口：6379
- 默认密码：dev123456
- 数据库：0

### 3. FireRedASR Service

**功能**: 自动语音识别服务

**技术细节**:
- 框架：FastAPI
- 模型：FireRedASR-AED-L + PUNC-BERT
- 支持语言：中文、英文
- API 端点：`POST /audio/transcriptions`

**核心代码**: `fireredasr-server/server/src/main.py`

**模型下载**:
```bash
cd server
mkdir -p models
git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L
```

### 4. FireRedTTS Service

**功能**: 文本转语音服务

**技术细节**:
- 模型：FireRedTTS-1S (revision: fireredtts1s_4_chat)
- API 端点：`POST /v1/audio/speech`
- 支持格式：MP3, WAV 等

**注意**: 该模型权重仅限非商业用途

**语音 ID**: `f531` (默认)

### 5. AI Agent Worker

**功能**: 核心 AI 代理逻辑

**核心文件**: `agents/examples/fireredchat_worker.py`

**关键组件**:
- `Agent`: AI 代理基类
- `AgentSession`: 会话管理器
- `RoomIO`: 房间输入输出处理
- `VAD`: 语音活动检测
- `Turn Detection`: 轮次检测

**角色配置**:
```python
character = {
    "nana": "塔罗占卜师角色配置...",
    "youyou": "AI 助手角色配置..."
}
```

### 6. Agents Playground (WebUI)

**功能**: 用户交互界面

**技术栈**:
- Next.js
- TypeScript
- Tailwind CSS
- LiveKit Client SDK

**核心组件**:
- `Playground.tsx`: 主界面
- `ChatTile.tsx`: 聊天窗口
- `ChatMessage.tsx`: 消息组件

**启动端口**: 3000

---

## 部署指南

### 前置要求

- Docker & Docker Compose
- NVIDIA GPU (用于 ASR/TTS 模型推理)
- CUDA 11.8 或 12.4
- 至少 16GB RAM
- 50GB 可用磁盘空间

### 步骤 1: 克隆项目

```bash
git clone --recurse-submodules https://github.com/FireRedTeam/FireRedChat.git
cd FireRedChat
```

### 步骤 2: 启动基础服务

```bash
cd docker
docker-compose up -d
```

**启动的服务**:
- Redis (6379)
- LiveKit RTC (7880, 7881, 50000-60000)
- WebUI (3000)

### 步骤 3: 部署 ASR 服务

```bash
cd fireredasr-server/server

# 下载模型
mkdir -p models
git clone https://huggingface.co/FireRedTeam/FireRedChat-punc models/PUNC-BERT
git clone https://huggingface.co/hfl/chinese-lert-base models/PUNC-BERT/chinese-lert-base
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L models/FireRedASR-AED-L

# 构建镜像 (根据 GPU 调整 CUDA 版本)
# Ampere 及更新：pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# Volta (3080, V100): pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
docker build -t fireredasr-service .

# 运行容器
docker run -d \
  -p 8000:8000 \
  --gpus '"cuda:0"' \
  -v $(pwd)/models:/app/models \
  fireredasr-service
```

### 步骤 4: 部署 TTS 服务

```bash
cd fireredtts-server

# 下载模型 (注意：仅限非商业用途)
huggingface-cli download --resume-download --repo-type model \
  FireRedTeam/FireRedTTS-1S --revision fireredtts1s_4_chat \
  --local-dir ./tts_4_chat

# 运行服务
docker run -td --name ttsserver \
  --security-opt seccomp:unconfined \
  -v "$(pwd)/tts_4_chat/pretrained_models:/workspace/models/redtts" \
  -p 8081:8081 \
  crpi-byegxpnnsibfy3j1.cn-shanghai.personal.cr.aliyuncs.com/fireredchat/fireredtts1-server:latest \
  bash /workspace/run.sh --llm --svc_config_path /workspace/svc.yaml --port 8081 --http_uri=/v1/audio/speech
```

### 步骤 5: 部署 LLM 服务

**选项 A: Ollama (轻量级)**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run qwen2.5-7b
```

**选项 B: vLLM (高性能)**

```bash
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct
```

### 步骤 6: 配置并启动 Agent Worker

```bash
cd agents/examples

# 复制环境配置
cp .env.dev .env

# 编辑 fireredchat_worker.py
# 修改以下配置：
# - LLM base_url: http://localhost:11434/v1 (Ollama)
# - ASR base_url: http://localhost:8000
# - TTS base_url: http://localhost:8081/v1

# 安装依赖
cd ..
pip install -e .
cd fireredchat-plugins
pip install -e livekit-plugins-firered
pip install -e livekit-plugins-fireredchat-pvad
pip install -e livekit-plugins-fireredchat-turn-detector

# 下载模型文件
python3 fireredchat_worker.py download-files

# 启动 Agent
python3 fireredchat_worker.py dev
```

### 步骤 7: 访问 WebUI

浏览器打开：`http://localhost:3000`

---

## API 参考

### FireRedASR API

**端点**: `POST /audio/transcriptions`

**请求**:
```bash
curl -X POST \
  http://localhost:8000/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.wav"
```

**响应**:
```json
{
  "sentences": [{
    "confidence": 0.8,
    "text": "识别的文本内容"
  }],
  "wav_file": "audio.wav"
}
```

### FireRedTTS API

**端点**: `POST /v1/audio/speech`

**请求**:
```bash
curl http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "哈喽，你好呀～",
    "voice": "f531",
    "response_format": "mp3"
  }' \
  --output audio.mp3
```

**响应**: 音频二进制数据

### LiveKit API

**房间连接**:
```python
from livekit import rtc

room = rtc.Room()
await room.connect("ws://localhost:7880", token)
```

**RPC 调用**:
```python
# 注册 RPC 方法
@ctx.room.local_participant.register_rpc_method("new_conversation")
async def new_conversation(data: rtc.RpcInvocationData):
    session.interrupt()
    session.clear_user_turn()
```

---

## 代码结构

### 项目目录树

```
FireRedChat/
├── docker/                          # Docker 配置
│   ├── docker-compose.yaml          # 服务编排
│   ├── livekit.yaml                 # LiveKit 配置
│   └── redis/data/                  # Redis 数据持久化
│
├── agents/                          # AI Agent 核心代码
│   ├── livekit-agents/              # LiveKit Agents 框架
│   │   ├── livekit/agents/
│   │   │   ├── agent.py             # Agent 基类
│   │   │   ├── session.py           # AgentSession
│   │   │   ├── worker.py            # Worker 协调器
│   │   │   ├── job.py               # 任务管理
│   │   │   ├── llm/                 # LLM 集成
│   │   │   ├── stt/                 # STT 集成
│   │   │   ├── tts/                 # TTS 集成
│   │   │   ├── vad.py               # VAD 实现
│   │   │   └── voice/               # 语音处理
│   │   └── livekit-plugins/         # 插件系统
│   │
│   ├── fireredchat-plugins/         # FireRed 定制插件
│   │   ├── livekit-plugins-firered/
│   │   │   ├── stt.py               # FireRed ASR 集成
│   │   │   └── tts.py               # FireRed TTS 集成
│   │   ├── livekit-plugins-fireredchat-pvad/
│   │   │   └── vad.py               # pVAD 实现
│   │   └── livekit-plugins-fireredchat-turn-detector/
│   │       └── base.py              # 轮次检测模型
│   │
│   ├── examples/
│   │   ├── fireredchat_worker.py    # 主 Worker 入口
│   │   └── voice_agents/            # 示例 Agent
│   │
│   └── tests/                       # 测试套件
│
├── agents-playground/               # WebUI
│   ├── src/
│   │   ├── components/
│   │   │   ├── playground/          # 游戏场组件
│   │   │   ├── chat/                # 聊天组件
│   │   │   └── config/              # 配置组件
│   │   └── hooks/                   # React Hooks
│   ├── public/
│   └── package.json
│
├── fireredasr-server/               # ASR 服务
│   └── server/
│       ├── src/
│       │   ├── main.py              # FastAPI 入口
│       │   └── routes/
│       │       ├── fireredasr.py    # ASR 路由
│       │       └── model.py         # 模型加载
│       └── redpost/                 # 后处理模块
│           ├── models/
│           │   ├── redpost.py       # 标点模型
│           │   └── redpunc_bert.py  # BERT 实现
│           └── data/
│               └── token_dict.py    # 词表
│
├── fireredtts-server/               # TTS 服务
│   └── README.md
│
└── README.md                        # 项目说明
```

### 关键类说明

#### Agent 类

```python
class Agent:
    """AI 代理基类"""
    
    def __init__(
        self,
        instructions: str,      # 系统指令
        tools: list = [],       # 工具函数
        llm: LLM = None,        # LLM 实例
    ):
        pass
    
    async def on_enter(self):
        """进入会话时调用"""
        pass
```

#### AgentSession 类

```python
class AgentSession:
    """会话管理器"""
    
    def __init__(
        self,
        vad: VAD = None,           # 语音活动检测
        stt: STT = None,           # 语音识别
        llm: LLM = None,           # 语言模型
        tts: TTS = None,           # 语音合成
        turn_detection: TurnDetector = None,  # 轮次检测
    ):
        pass
    
    async def start(
        self,
        agent: Agent,
        room: rtc.Room,
        room_input_options: RoomInputOptions = None,
        room_output_options: RoomOutputOptions = None,
    ):
        """启动会话"""
        pass
    
    async def generate_reply(self, instructions: str = None):
        """生成回复"""
        pass
    
    async def interrupt(self):
        """中断当前回复"""
        pass
    
    async def clear_user_turn(self):
        """清除用户轮次"""
        pass
```

#### RoomIO 类

```python
class RoomIO:
    """房间输入输出处理器"""
    
    def __init__(self, session: AgentSession, room: rtc.Room):
        pass
    
    async def start(self):
        """启动 IO 流"""
        pass
```

---

## 配置说明

### 环境变量

#### Agent Worker (.env)

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
OPENAI_BASE_URL=http://localhost:11434/v1  # Ollama 兼容
OPENAI_API_KEY=notneeded
```

### 角色配置 (fireredchat_worker.py)

```python
character = {
    "nana": """
    简介：娜娜，塔罗占卜师
    性格：好奇、想象力丰富、神秘
    语言风格：浪漫比喻、诗意的语言
    开场白：哈喽，我是娜娜。
    """,
    
    "youyou": """
    简介：悠悠，AI 助手
    性格：霸气、酷、可爱、偏宠闺蜜
    语言风格：口语化、简洁
    开场白：哈喽，我是悠悠。
    """
}
```

### Docker 配置 (docker-compose.yaml)

```yaml
services:
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    environment:
      REDISCLI_AUTH: dev123456
  
  livekitrtc:
    image: livekit/livekit-server:latest
    ports:
      - "7880:7880"
      - "7881:7881"
      - "50000-60000:50000-60000"
    volumes:
      - ./livekit.yaml:/livekit.yaml:ro
  
  livekitplay:
    image: node:20-alpine
    ports:
      - "3000:3000"
    volumes:
      - ../agents-playground:/agents-playground
```

---

## 使用示例

### 示例 1: 创建自定义 Agent

```python
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import firered, openai, fireredchat_pvad

class CustomAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(
            instructions=instructions,
        )
    
    async def on_enter(self):
        self.session.generate_reply()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    agent = CustomAgent("你是一个友好的 AI 助手。")
    
    session = AgentSession(
        vad=fireredchat_pvad.VAD.load(),
        llm=openai.LLM.with_ollama(model="qwen2.5-7b"),
        stt=firered.STT(model="FireRedASR-AED-1"),
        tts=firered.TTS(voice="f531"),
    )
    
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### 示例 2: 使用工具函数

```python
from livekit.agents import function_tool, RunContext

@function_tool
async def get_weather(context: RunContext, location: str):
    """查询天气"""
    # 实现天气查询逻辑
    return {"weather": "晴朗", "temperature": 25}

agent = Agent(
    instructions="你是一个天气助手。",
    tools=[get_weather],
)
```

### 示例 3: 多 Agent 切换

```python
class IntroAgent(Agent):
    async def on_enter(self):
        self.session.generate_reply()
    
    @function_tool
    async def start_story(self, context: RunContext):
        """开始讲故事"""
        story_agent = StoryAgent()
        return story_agent, "让我们开始故事吧！"

class StoryAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个讲故事的人。"
        )
```

### 示例 4: 保存对话记录

```python
from datetime import datetime
import json

async def save_transcript(session, room_name):
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/workspace/logs/transcript_{room_name}_{current_date}.json"
    
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(session.history.to_dict(), f, indent=4, ensure_ascii=False)
    
    print(f"对话记录已保存到：{filename}")

# 在 shutdown 回调中调用
ctx.add_shutdown_callback(lambda: save_transcript(session, ctx.room.name))
```

---

## 故障排查

### 常见问题

#### 1. Agent Worker 无法连接 LiveKit

**症状**: `Connection refused` 错误

**解决方案**:
```bash
# 检查 LiveKit 服务状态
docker ps | grep livekitrtc

# 查看日志
docker logs livekitrtc

# 确认端口开放
netstat -tlnp | grep 7880
```

#### 2. ASR 服务响应慢

**症状**: 语音识别延迟高

**解决方案**:
- 确认 GPU 可用：`nvidia-smi`
- 检查模型是否完整下载
- 调整 CUDA 版本匹配 GPU

#### 3. TTS 服务返回错误

**症状**: `500 Internal Server Error`

**解决方案**:
```bash
# 检查 TTS 容器日志
docker logs ttsserver

# 确认模型路径正确
ls -la tts_4_chat/pretrained_models/

# 重启服务
docker restart ttsserver
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
```

#### 5. 语音识别不准确

**解决方案**:
- 检查音频质量 (建议 16kHz, 16bit)
- 确认 PUNC-BERT 模型正确加载
- 调整 VAD 阈值

### 日志查看

```bash
# Agent Worker 日志
tail -f /workspace/logs/*.log

# Docker 服务日志
docker logs -f livekitrtc
docker logs -f ttsserver
docker logs -f fireredasr

# Redis 监控
docker exec -it <redis_container> redis-cli monitor
```

---

## 开发指南

### 添加新插件

1. **创建插件目录**:
```bash
cd agents/fireredchat-plugins
mkdir livekit-plugins-myplugin
```

2. **实现插件接口**:
```python
# livekit/plugins/myplugin/__init__.py
from .stt import STT
from .version import __version__

__all__ = ["STT", "__version__"]
```

3. **注册插件**:
```python
# setup.py
from setuptools import setup

setup(
    name="livekit-plugins-myplugin",
    version="0.1.0",
    packages=["livekit.plugins.myplugin"],
)
```

### 自定义 VAD 模型

```python
from livekit.plugins.fireredchat_pvad import VAD

# 调整参数
vad = VAD.load(
    activation_threshold=0.5,  # 激活阈值
    min_speech_duration=0.25,  # 最小语音时长
    min_silence_duration=0.5,  # 最小静音时长
)
```

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 运行开发模式
python3 fireredchat_worker.py dev

# 使用 Python 调试器
python3 -m pdb fireredchat_worker.py dev
```

### 性能优化

1. **GPU 内存优化**:
```python
# 限制 GPU 内存使用
import torch
torch.cuda.set_per_process_memory_fraction(0.8)
```

2. **批处理推理**:
```python
# ASR 批处理
stt = firered.STT(batch_size=4)
```

3. **缓存优化**:
```python
# 启用 LLM 缓存
llm = openai.LLM.with_ollama(
    model="qwen2.5-7b",
    cache_dir="/workspace/cache"
)
```

### 测试

```bash
# 运行单元测试
cd agents
pytest tests/

# 运行特定测试
pytest tests/test_stt.py -v

# 覆盖率报告
pytest --cov=livekit.agents tests/
```

---

## 附录

### 相关资源

- **官方文档**: https://docs.livekit.io/agents/
- **GitHub 仓库**: https://github.com/FireRedTeam/FireRedChat
- **Demo 页面**: https://fireredteam.github.io/demos/firered_chat/
- **论文**: https://arxiv.org/pdf/2509.06502
- **HuggingFace**: https://huggingface.co/FireRedTeam

### 许可证

- 主项目：Apache 2.0
- FireRedTTS 模型：仅限非商业用途

### 致谢

- [LiveKit](https://github.com/livekit/livekit) - RTC 框架
- [LiveKit Agents](https://github.com/livekit/agents) - Agent 框架
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - 语音模型
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - NLP 模型

---

_文档结束_

