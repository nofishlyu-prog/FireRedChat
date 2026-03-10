# FireRedChat API 参考手册

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10  
> 🎯 版本：v1.0

---

## 📑 目录

1. [API 概览](#api-概览)
2. [FireRedASR API](#fireredasr-api)
3. [FireRedTTS API](#fireredtts-api)
4. [LiveKit API](#livekit-api)
5. [Agent Worker API](#agent-worker-api)
6. [错误码参考](#错误码参考)
7. [使用示例](#使用示例)

---

## API 概览

### 服务端点

| 服务 | 基础 URL | 默认端口 |
|------|----------|----------|
| FireRedASR | `http://localhost:8000` | 8000 |
| FireRedTTS | `http://localhost:8081` | 8081 |
| LiveKit | `ws://localhost:7880` | 7880 |
| WebUI | `http://localhost:3000` | 3000 |

### 认证方式

```bash
# ASR/TTS: 无需认证 (内网部署)
# 生产环境建议添加 API Key

# LiveKit: Token 认证
# 使用 JWT 生成访问令牌
```

### 数据格式

- **请求**: JSON / multipart/form-data
- **响应**: JSON / 音频二进制
- **编码**: UTF-8
- **音频格式**: WAV, MP3, PCM16

---

## FireRedASR API

### 语音识别

**端点**: `POST /audio/transcriptions`

**描述**: 将音频文件转换为文本

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | binary | ✅ | 音频文件 (WAV/MP3) |

**请求示例**:

```bash
curl -X POST \
  http://localhost:8000/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@test.wav"
```

**响应格式**:

```json
{
  "sentences": [
    {
      "confidence": 0.95,
      "text": "识别的文本内容"
    }
  ],
  "wav_file": "test.wav"
}
```

**响应字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `sentences` | array | 句子列表 |
| `sentences[].confidence` | float | 置信度 (0-1) |
| `sentences[].text` | string | 识别文本 |
| `wav_file` | string | 文件名 |

**Python 示例**:

```python
import requests

files = {'file': open('test.wav', 'rb')}
response = requests.post(
    'http://localhost:8000/audio/transcriptions',
    files=files
)

result = response.json()
print(result['sentences'][0]['text'])
```

**Node.js 示例**:

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('test.wav'));

const response = await axios.post(
  'http://localhost:8000/audio/transcriptions',
  form,
  { headers: form.getHeaders() }
);

console.log(response.data.sentences[0].text);
```

---

## FireRedTTS API

### 语音合成

**端点**: `POST /v1/audio/speech`

**描述**: 将文本合成为语音

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input` | string | ✅ | 输入文本 |
| `voice` | string | ✅ | 语音 ID |
| `model` | string | ❌ | 模型名称 (默认：fireredtts1.0) |
| `response_format` | string | ❌ | 输出格式 (mp3/wav) |
| `speed` | float | ❌ | 语速 (0.5-2.0, 默认：1.0) |
| `instructions` | string | ❌ | 合成指令 |

**语音 ID 列表**:

| ID | 性别 | 风格 | 适用场景 |
|----|------|------|----------|
| f531 | 女 | 温柔、亲切 | 通用助手 |
| f532 | 女 | 活泼、可爱 | 聊天机器人 |
| f533 | 女 | 专业、正式 | 新闻播报 |
| m501 | 男 | 沉稳、可靠 | 商务助手 |
| m502 | 男 | 阳光、开朗 | 娱乐应用 |

**请求示例**:

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

**响应**: 音频二进制数据 (MP3/WAV)

**Python 示例**:

```python
import requests

response = requests.post(
    'http://localhost:8081/v1/audio/speech',
    json={
        'input': '哈喽，你好呀～',
        'voice': 'f531',
        'response_format': 'mp3'
    }
)

with open('audio.mp3', 'wb') as f:
    f.write(response.content)
```

**Node.js 示例**:

```javascript
const axios = require('axios');
const fs = require('fs');

const response = await axios.post(
  'http://localhost:8081/v1/audio/speech',
  {
    input: '哈喽，你好呀～',
    voice: 'f531',
    response_format: 'mp3'
  },
  {
    responseType: 'arraybuffer'
  }
);

fs.writeFileSync('audio.mp3', response.data);
```

**流式合成**:

```python
import requests

response = requests.post(
    'http://localhost:8081/v1/audio/speech',
    json={
        'input': '这是一段长文本',
        'voice': 'f531',
        'stream': True
    },
    stream=True
)

with open('audio.mp3', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

---

## LiveKit API

### 房间连接

**端点**: WebSocket `ws://localhost:7880`

**描述**: 连接到 LiveKit 房间

**认证 Token 生成**:

```python
from livekit import api

# 生成访问令牌
token = api.AccessToken(
    api_key="devkey",
    api_secret="devsecret"
).with_identity("user123").with_name("User").to_jwt()
```

**连接示例**:

```python
from livekit import rtc

async def connect_to_room():
    room = rtc.Room()
    
    # 连接房间
    await room.connect(
        "ws://localhost:7880",
        token
    )
    
    # 发布音频轨道
    track = rtc.LocalAudioTrack.create_audio_track(
        "audio",
        audio_source
    )
    
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    
    await room.local_participant.publish_track(track, options)
```

### RPC 调用

**端点**: `POST /rpc/{method}`

**描述**: 调用房间 RPC 方法

**注册 RPC 方法**:

```python
@ctx.room.local_participant.register_rpc_method("new_conversation")
async def new_conversation(data: rtc.RpcInvocationData):
    """重置对话"""
    session.interrupt()
    session.clear_user_turn()
    await session._agent.update_chat_ctx(ChatContext.empty())
    
    return {"status": "ok"}
```

**调用 RPC**:

```python
result = await room.local_participant.perform_rpc(
    destination_identity="agent",
    method="new_conversation",
    payload=json.dumps({})
)
```

---

## Agent Worker API

### 配置 Agent

**文件**: `agents/examples/fireredchat_worker.py`

**角色配置**:

```python
character = {
    "youyou": """
    你是悠悠，一个有灵魂的大语言模型助手。
    性格：霸气、酷、可爱、偏宠闺蜜。
    语言风格：口语化、简洁。
    开场白：哈喽，我是悠悠。
    """,
    
    "nana": """
    你是娜娜，一名塔罗占卜师。
    性格：好奇、想象力丰富、神秘。
    语言风格：浪漫比喻、诗意的语言。
    开场白：哈喽，我是娜娜。
    """
}
```

### 自定义 Agent

**创建自定义 Agent**:

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

### 添加工具函数

```python
from livekit.agents import function_tool, RunContext

@function_tool
async def get_weather(context: RunContext, location: str):
    """查询天气"""
    # 实现天气查询逻辑
    return {"weather": "晴朗", "temperature": 25}

@function_tool
async def search_web(context: RunContext, query: str):
    """搜索网络"""
    # 实现搜索逻辑
    return {"results": [...]}

# 注册到 Agent
agent = Agent(
    instructions="你是一个天气助手。",
    tools=[get_weather, search_web],
)
```

---

## 错误码参考

### FireRedASR 错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|-----------|------|
| `INVALID_AUDIO` | 400 | 音频格式无效 |
| `AUDIO_TOO_LONG` | 400 | 音频超过最大时长 |
| `MODEL_NOT_LOADED` | 503 | 模型未加载完成 |
| `GPU_OUT_OF_MEMORY` | 503 | GPU 内存不足 |
| `INFERENCE_FAILED` | 500 | 推理失败 |

**错误响应示例**:

```json
{
  "error": {
    "code": "INVALID_AUDIO",
    "message": "Unsupported audio format. Expected WAV or MP3.",
    "details": {
      "received_format": "flac",
      "supported_formats": ["wav", "mp3"]
    }
  }
}
```

### FireRedTTS 错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|-----------|------|
| `INVALID_TEXT` | 400 | 文本格式无效 |
| `TEXT_TOO_LONG` | 400 | 文本超过最大长度 |
| `INVALID_VOICE` | 400 | 语音 ID 不存在 |
| `SYNTHESIS_FAILED` | 500 | 合成失败 |

### LiveKit 错误码

| 错误码 | 说明 |
|--------|------|
| `TOKEN_EXPIRED` | Token 已过期 |
| `ROOM_NOT_FOUND` | 房间不存在 |
| `PERMISSION_DENIED` | 权限不足 |
| `CONNECTION_FAILED` | 连接失败 |

---

## 使用示例

### 完整对话流程

```python
import requests
from livekit import rtc, api

class VoiceChatBot:
    def __init__(self):
        self.asr_url = "http://localhost:8000"
        self.tts_url = "http://localhost:8081"
        self.llm_url = "http://localhost:11434"
    
    async def process_user_audio(self, audio_data):
        """处理用户音频"""
        # 1. ASR: 语音→文本
        asr_response = requests.post(
            f"{self.asr_url}/audio/transcriptions",
            files={'file': audio_data}
        )
        user_text = asr_response.json()['sentences'][0]['text']
        
        # 2. LLM: 生成回复
        llm_response = requests.post(
            f"{self.llm_url}/api/generate",
            json={
                "model": "qwen2.5-7b",
                "prompt": user_text,
                "stream": False
            }
        )
        bot_text = llm_response.json()['response']
        
        # 3. TTS: 文本→语音
        tts_response = requests.post(
            f"{self.tts_url}/v1/audio/speech",
            json={
                "input": bot_text,
                "voice": "f531",
                "response_format": "mp3"
            }
        )
        bot_audio = tts_response.content
        
        return bot_audio

# 使用示例
bot = VoiceChatBot()
response = await bot.process_user_audio(audio_data)
```

### 批量转录

```python
import requests
from pathlib import Path

def batch_transcribe(audio_dir, output_file):
    """批量转录音频文件"""
    results = []
    
    for audio_path in Path(audio_dir).glob("*.wav"):
        print(f"Processing {audio_path}...")
        
        with open(audio_path, 'rb') as f:
            response = requests.post(
                'http://localhost:8000/audio/transcriptions',
                files={'file': f}
            )
        
        result = response.json()
        results.append({
            'file': str(audio_path),
            'text': result['sentences'][0]['text'],
            'confidence': result['sentences'][0]['confidence']
        })
    
    # 保存结果
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

# 使用示例
results = batch_transcribe('./recordings', './transcripts.json')
```

### 语音克隆

```python
import requests

def clone_voice(reference_audio, target_text, output_file):
    """
    语音克隆 (需要 TTS 服务支持)
    
    Args:
        reference_audio: 参考音频路径 (5-10 秒)
        target_text: 要合成的文本
        output_file: 输出音频路径
    """
    # 1. 上传参考音频获取声纹
    with open(reference_audio, 'rb') as f:
        voice_response = requests.post(
            'http://localhost:8081/v1/voices/clone',
            files={'audio': f}
        )
    
    voice_id = voice_response.json()['voice_id']
    
    # 2. 使用克隆的语音合成
    tts_response = requests.post(
        'http://localhost:8081/v1/audio/speech',
        json={
            'input': target_text,
            'voice': voice_id,
            'response_format': 'mp3'
        }
    )
    
    # 3. 保存结果
    with open(output_file, 'wb') as f:
        f.write(tts_response.content)
    
    return voice_id

# 使用示例
voice_id = clone_voice(
    'reference.wav',
    '这是用克隆语音合成的文本',
    'output.mp3'
)
```

### 实时流式转录

```python
import pyaudio
import requests
import threading

class StreamingASR:
    def __init__(self):
        self.asr_url = "http://localhost:8000"
        self.session_id = None
    
    def start_session(self):
        """开始流式会话"""
        response = requests.post(f"{self.asr_url}/stream/start")
        self.session_id = response.json()['session_id']
        return self.session_id
    
    def send_audio(self, audio_chunk):
        """发送音频块"""
        response = requests.post(
            f"{self.asr_url}/stream/{self.session_id}/audio",
            data=audio_chunk
        )
        return response.json()
    
    def get_result(self):
        """获取最终结果"""
        response = requests.post(
            f"{self.asr_url}/stream/{self.session_id}/end"
        )
        return response.json()

# 使用示例
asr = StreamingASR()
asr.start_session()

# 从麦克风录制并发送
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)

try:
    while True:
        audio_chunk = stream.read(1600)  # 100ms
        result = asr.send_audio(audio_chunk)
        
        if 'interim' in result:
            print(f"Interim: {result['interim']}")
        
        if 'final' in result:
            print(f"Final: {result['final']}")
finally:
    stream.stop_stream()
    stream.close()
    asr.get_result()
```

### Webhook 回调

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/asr-complete', methods=['POST'])
def asr_complete():
    """ASR 完成回调"""
    data = request.json
    
    print(f"Transcription complete:")
    print(f"  File: {data['wav_file']}")
    print(f"  Text: {data['sentences'][0]['text']}")
    print(f"  Confidence: {data['sentences'][0]['confidence']}")
    
    return jsonify({"status": "ok"})

@app.route('/webhook/tts-complete', methods=['POST'])
def tts_complete():
    """TTS 完成回调"""
    data = request.json
    
    print(f"Synthesis complete:")
    print(f"  Text: {data['input']}")
    print(f"  Voice: {data['voice']}")
    print(f"  Duration: {data['duration']}s")
    
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(port=5000)
```

---

## 附录：SDK 参考

### Python SDK

```bash
# 安装
pip install livekit-agents
pip install firered-plugins
```

**快速开始**:

```python
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import firered, openai, fireredchat_pvad

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    agent = Agent(instructions="你是悠悠，一个 AI 助手。")
    
    session = AgentSession(
        vad=fireredchat_pvad.VAD.load(),
        llm=openai.LLM.with_ollama(model="qwen2.5-7b"),
        stt=firered.STT(),
        tts=firered.TTS(voice="f531"),
    )
    
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Node.js SDK

```bash
# 安装
npm install livekit-client
```

**快速开始**:

```javascript
import { Room, RoomEvent } from 'livekit-client';

async function connectToRoom(token, url) {
  const room = new Room();
  
  await room.connect(url, token);
  
  room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    console.log('Track subscribed:', track.kind);
  });
  
  return room;
}
```

---

_API 参考手册结束_
