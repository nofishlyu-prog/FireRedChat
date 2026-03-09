# FireRedChat 代码结构详解

> 🔍 深入理解项目代码组织

---

## 📁 顶层目录

```
FireRedChat/
├── docker/                    # Docker 部署配置
├── agents/                    # AI Agent 核心代码 (fork of livekit/agents)
├── agents-playground/         # WebUI (fork of livekit/agents-playground)
├── fireredasr-server/         # ASR 语音识别服务
├── fireredtts-server/         # TTS 语音合成服务
├── PROJECT_DOCUMENTATION.md   # 完整项目文档
├── QUICKSTART_ZH.md          # 快速入门指南
└── README.md                  # 项目说明
```

---

## 🐳 docker/ - 部署配置

```
docker/
├── docker-compose.yaml        # 服务编排配置
├── livekit.yaml               # LiveKit 服务器配置
└── redis/data/                # Redis 数据持久化目录
```

### 关键配置

**docker-compose.yaml** 定义 3 个核心服务:
- `redis`: 状态存储 (端口 6379)
- `livekitrtc`: RTC 服务器 (端口 7880, 7881, 50000-60000)
- `livekitplay`: WebUI (端口 3000)

**livekit.yaml** 配置:
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

---

## 🤖 agents/ - AI Agent 核心

```
agents/
├── livekit-agents/                    # LiveKit Agents 框架
│   ├── livekit/agents/
│   │   ├── __init__.py                # 公共导出
│   │   ├── agent.py                   # Agent 基类 ⭐
│   │   ├── session.py                 # AgentSession ⭐
│   │   ├── worker.py                  # Worker 协调器 ⭐
│   │   ├── job.py                     # 任务管理
│   │   ├── types.py                   # 类型定义
│   │   ├── vad.py                     # VAD 接口
│   │   │
│   │   ├── cli/                       # 命令行工具
│   │   │   ├── cli.py                 # CLI 入口
│   │   │   └── dev.py                 # 开发模式
│   │   │
│   │   ├── ipc/                       # 进程间通信
│   │   │   ├── inference_pool.py      # 推理池
│   │   │   └── supervised_proc.py     # 监督进程
│   │   │
│   │   ├── llm/                       # LLM 集成
│   │   │   ├── llm.py                 # LLM 基类
│   │   │   ├── chat_ctx.py            # 聊天上下文
│   │   │   └── function_tool.py       # 工具函数
│   │   │
│   │   ├── stt/                       # 语音识别
│   │   │   ├── stt.py                 # STT 基类
│   │   │   └── stream_adapter.py      # 流适配器
│   │   │
│   │   ├── tts/                       # 语音合成
│   │   │   ├── tts.py                 # TTS 基类
│   │   │   └── chunked_stream.py      # 分块流
│   │   │
│   │   ├── tokenize/                  # 分词器
│   │   │   ├── tokenizer.py           # 分词器基类
│   │   │   └── basic.py               # 基础分词
│   │   │
│   │   ├── voice/                     # 语音处理
│   │   │   ├── agent_output.py        # Agent 输出
│   │   │   ├── speech_handle.py       # 语音处理
│   │   │   └── turn_tracker.py        # 轮次追踪
│   │   │
│   │   ├── utils/                     # 工具函数
│   │   │   ├── aio.py                 # 异步工具
│   │   │   ├── audio.py               # 音频处理
│   │   │   └── http.py                # HTTP 工具
│   │   │
│   │   ├── metrics/                   # 指标收集
│   │   └── telemetry/                 # 遥测数据
│   │
│   ├── livekit-plugins/               # 官方插件
│   │   ├── livekit-plugins-openai/    # OpenAI 集成
│   │   ├── livekit-plugins-silero/    # Silero VAD
│   │   ├── livekit-plugins-deepgram/  # Deepgram STT
│   │   └── ...
│   │
│   ├── fireredchat-plugins/           # FireRed 定制插件 ⭐
│   │   ├── livekit-plugins-firered/
│   │   │   ├── __init__.py
│   │   │   ├── stt.py                 # FireRed ASR 集成 ⭐
│   │   │   └── tts.py                 # FireRed TTS 集成 ⭐
│   │   │
│   │   ├── livekit-plugins-fireredchat-pvad/
│   │   │   ├── __init__.py
│   │   │   ├── vad.py                 # pVAD 实现 ⭐
│   │   │   └── log.py
│   │   │
│   │   └── livekit-plugins-fireredchat-turn-detector/
│   │       ├── __init__.py
│   │       └── base.py                # 中文轮次检测 ⭐
│   │
│   ├── examples/                      # 示例代码
│   │   ├── fireredchat_worker.py      # FireRed 主 Worker ⭐⭐⭐
│   │   ├── minimal_worker.py          # 最小示例
│   │   ├── voice_agents/              # 语音 Agent 示例
│   │   │   ├── basic_agent.py
│   │   │   ├── multi_agent.py
│   │   │   └── structured_output.py
│   │   └── other/                     # 其他示例
│   │
│   └── tests/                         # 测试套件
│       ├── test_agent_session.py
│       ├── test_stt.py
│       ├── test_tts.py
│       └── ...
```

### 核心类说明

#### Agent (agent.py)
```python
class Agent:
    """AI 代理基类
    
    属性:
        instructions: 系统指令
        tools: 可用工具函数列表
        llm: LLM 实例
        chat_ctx: 聊天上下文
    
    方法:
        on_enter(): 进入会话时调用
        on_exit(): 退出会话时调用
    """
```

#### AgentSession (session.py)
```python
class AgentSession:
    """会话管理器 - 核心中的核心
    
    管理:
        - 语音识别流
        - LLM 推理
        - 语音合成流
        - VAD 检测
        - 轮次检测
    
    关键方法:
        start(): 启动会话
        generate_reply(): 生成回复
        interrupt(): 中断当前回复
        clear_user_turn(): 清除用户轮次
    """
```

#### Worker (worker.py)
```python
class Worker:
    """Worker 协调器
    
    职责:
        - 连接 LiveKit 服务器
        - 接收任务分配
        - 启动 Agent 进程
        - 管理资源
    
    配置:
        WorkerOptions(
            entrypoint_fnc=entrypoint,  # 入口函数
            prewarm_fnc=prewarm,        # 预热函数
            job_memory_warn_mb=1500,    # 内存警告阈值
        )
    """
```

---

## 🎨 agents-playground/ - WebUI

```
agents-playground/
├── src/
│   ├── components/
│   │   ├── playground/
│   │   │   ├── Playground.tsx           # 主界面 ⭐
│   │   │   ├── PlaygroundHeader.tsx     # 头部
│   │   │   ├── PlaygroundTile.tsx       # 内容块
│   │   │   ├── SettingsDropdown.tsx     # 设置下拉
│   │   │   └── PlaygroundDeviceSelector.tsx
│   │   │
│   │   ├── chat/
│   │   │   ├── ChatTile.tsx             # 聊天窗口 ⭐
│   │   │   ├── ChatMessage.tsx          # 消息组件
│   │   │   └── ChatMessageInput.tsx     # 输入框
│   │   │
│   │   ├── config/
│   │   │   ├── ConfigurationPanelItem.tsx
│   │   │   ├── NameValueRow.tsx
│   │   │   └── AudioInputTile.tsx
│   │   │
│   │   └── toast/
│   │       ├── ToasterProvider.tsx
│   │       └── PlaygroundToast.tsx
│   │
│   ├── hooks/
│   │   ├── useLiveKit.ts                # LiveKit Hook ⭐
│   │   └── ...
│   │
│   └── App.tsx                          # 应用入口
│
├── public/
├── package.json
├── next.config.js
├── tailwind.config.js
└── tsconfig.json
```

### 关键组件

**Playground.tsx**: 主界面
- 连接 LiveKit 房间
- 管理音视频设备
- 渲染聊天界面

**ChatTile.tsx**: 聊天窗口
- 显示消息历史
- 实时转录显示
- 消息输入

**useLiveKit.ts**: LiveKit Hook
- 房间连接
- 参与者管理
- 数据通道

---

## 🎤 fireredasr-server/ - ASR 服务

```
fireredasr-server/
└── server/
    ├── Dockerfile
    ├── requirements.txt
    └── src/
        ├── main.py                      # FastAPI 入口 ⭐
        ├── helpers.py                   # 辅助函数
        └── routes/
            ├── fireredasr.py            # ASR 路由 ⭐
            └── model.py                 # 模型加载 ⭐
        │
        └── redpost/                     # 后处理模块
            ├── __init__.py
            ├── models/
            │   ├── redpost.py           # 标点恢复模型 ⭐
            │   └── redpunc_bert.py      # BERT 实现
            └── data/
                ├── token_dict.py        # 词表
                └── hf_bert_tokenizer.py # BERT 分词器
```

### 核心流程

**main.py**:
```python
app = FastAPI()

@app.post("/audio/transcriptions")
async def transcribe(file: UploadFile):
    # 1. 加载音频
    # 2. 调用 FireRedASR 模型
    # 3. 应用标点模型 (redpost)
    # 4. 返回结果
    pass
```

**redpost.py**:
```python
class RedPostModel:
    """标点恢复模型
    
    输入：无标点文本
    输出：带标点文本
    
    模型：BERT-base-multilingual-cased
    """
```

---

## 🔊 fireredtts-server/ - TTS 服务

```
fireredtts-server/
├── README.md
└── (Docker 镜像配置)
```

**注意**: TTS 服务使用预构建的 Docker 镜像，源码未公开。

API 端点: `POST /v1/audio/speech`

---

## 📊 数据流图

```
用户语音
    │
    ▼
┌─────────────────┐
│  LiveKit RTC    │
│  (WebRTC 流)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RoomIO        │
│  (音频提取)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   pVAD          │
│  (语音检测)      │
└────────┬────────┘
         │ 语音段
         ▼
┌─────────────────┐
│  FireRedASR     │
│  (语音→文本)     │
└────────┬────────┘
         │ 文本
         ▼
┌─────────────────┐
│   LLM           │
│  (文本→回复)     │
└────────┬────────┘
         │ 回复文本
         ▼
┌─────────────────┐
│  FireRedTTS     │
│  (文本→语音)     │
└────────┬────────┘
         │ 音频
         ▼
┌─────────────────┐
│  LiveKit RTC    │
│  (发送给用户)    │
└─────────────────┘
```

---

## 🔑 关键文件索引

| 文件 | 作用 | 重要性 |
|------|------|--------|
| `agents/examples/fireredchat_worker.py` | 主 Worker 入口 | ⭐⭐⭐ |
| `agents/livekit-agents/livekit/agents/agent.py` | Agent 基类 | ⭐⭐ |
| `agents/livekit-agents/livekit/agents/session.py` | 会话管理 | ⭐⭐⭐ |
| `agents/livekit-agents/livekit/agents/worker.py` | Worker 协调 | ⭐⭐ |
| `agents/fireredchat-plugins/livekit-plugins-firered/stt.py` | ASR 集成 | ⭐⭐ |
| `agents/fireredchat-plugins/livekit-plugins-firered/tts.py` | TTS 集成 | ⭐⭐ |
| `agents/fireredchat-plugins/livekit-plugins-fireredchat-pvad/vad.py` | pVAD 实现 | ⭐⭐ |
| `agents/fireredchat-plugins/livekit-plugins-fireredchat-turn-detector/base.py` | 轮次检测 | ⭐⭐ |
| `fireredasr-server/server/src/main.py` | ASR 服务入口 | ⭐⭐ |
| `fireredasr-server/server/src/routes/fireredasr.py` | ASR API 实现 | ⭐ |
| `fireredasr-server/server/redpost/models/redpost.py` | 标点恢复 | ⭐ |
| `docker/docker-compose.yaml` | 服务编排 | ⭐⭐ |
| `agents-playground/src/components/playground/Playground.tsx` | WebUI 主界面 | ⭐⭐ |

---

## 📝 修改指南

### 修改 Agent 行为

编辑：`agents/examples/fireredchat_worker.py`

```python
# 1. 修改角色指令
character = {
    "youyou": "新的角色描述..."
}

# 2. 修改 LLM 配置
used_llm = openai.LLM.with_ollama(
    model="your-model",
    base_url="http://your-llm-server:11434/v1",
)

# 3. 修改 VAD 参数
vad=fireredchat_pvad.VAD.load(activation_threshold=0.5)
```

### 添加新工具

```python
from livekit.agents import function_tool, RunContext

@function_tool
async def my_tool(context: RunContext, arg1: str):
    """工具描述"""
    # 实现逻辑
    return {"result": "value"}

# 注册到 Agent
agent = Agent(
    instructions="...",
    tools=[my_tool],
)
```

### 自定义 WebUI

编辑：`agents-playground/src/components/playground/Playground.tsx`

```tsx
// 修改界面布局
// 添加新组件
// 调整样式
```

---

_代码结构文档结束_
