# FireRedChat 代码逻辑详解

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10  
> 🎯 目标：深入解释各模块代码的详细逻辑

---

## 📑 目录

1. [核心架构概览](#核心架构概览)
2. [Agent Worker 入口](#agent-worker-入口)
3. [Agent 基类详解](#agent-基类详解)
4. [会话管理 (AgentSession)](#会话管理)
5. [语音活动检测 (pVAD)](#语音活动检测-pvad)
6. [轮次检测 (Turn Detector)](#轮次检测-turn-detector)
7. [ASR 集成](#asr-集成)
8. [TTS 集成](#tts-集成)
9. [数据流全景](#数据流全景)
10. [关键配置参数](#关键配置参数)

---

## 核心架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    用户浏览器 (WebUI)                        │
│                  agents-playground:3000                      │
└────────────────────┬────────────────────────────────────────┘
                     │ WebSocket (LiveKit RTC)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  LiveKit RTC Server                          │
│                  Port: 7880 (信令) / 50000-60000 (媒体)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   AI Agent Worker                            │
│              (fireredchat_worker.py 入口)                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   RoomIO     │→ │ AgentSession │→ │    Agent     │       │
│  │  (房间 IO)    │  │  (会话管理)   │  │  (AI 代理)     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                │                  │                │
│         ▼                ▼                  ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Audio Track │  │  VAD + STT   │  │  LLM + TTS   │       │
│  │  (音频轨道)   │  │  (语音检测)   │  │  (回复生成)   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ FireRed  │ │ FireRed  │ │   LLM    │
   │   ASR    │ │   TTS    │ │ (Ollama) │
   │ :8000    │ │ :8081    │ │ :11434   │
   └──────────┘ └──────────┘ └──────────┘
```

---

## Agent Worker 入口

**文件**: `agents/examples/fireredchat_worker.py`

这是整个系统的**启动入口**，负责：
1. 初始化 Worker 进程
2. 配置 Agent 角色
3. 连接 LiveKit 房间
4. 启动会话管理

### 核心代码解析

```python
# ============ 1. 角色定义 ============
character = {
    "nana": "简介：娜娜，塔罗占卜师...",
    "youyou": "简介：悠悠，AI 助手..."
}

# ============ 2. Agent 类定义 ============
class MyAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(
            instructions=instructions,  # 角色指令
            chat_ctx=ChatContext.empty()  # 空聊天上下文
        )

    async def on_enter(self):
        # 进入会话时自动生成回复
        self.session.generate_reply()

# ============ 3. Worker 预热 ============
def prewarm(proc: JobProcess):
    """预加载 VAD 模型到进程缓存"""
    proc.userdata["vad"] = fireredchat_pvad.VAD.load(
        activation_threshold=0.5  # 语音激活阈值
    )

# ============ 4. Worker 入口函数 ============
async def entrypoint(ctx: JobContext):
    # 从房间名解析场景
    scene = ctx.room.name.rsplit("-", 1)[-1]
    name = "youyou"  # 固定使用悠悠角色
    
    # 创建 Agent 实例
    instruction = character.get(name, character["nana"])
    used_agent = MyAgent(instruction)
    
    # 配置 LLM (使用 Ollama)
    used_llm = openai.LLM.with_ollama(
        model="qwen2.5-7b",
        base_url="http://localhost:11434/v1",
    )
    
    # 创建会话管理器
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],  # 预加载的 VAD
        llm=used_llm,                   # LLM
        stt=firered.STT(                # FireRed ASR
            model="FireRedASR-AED-1",
            base_url="http://localhost:8000",
        ),
        tts=firered.TTS(                # FireRed TTS
            model="fireredtts1.0",
            voice="f531",
            base_url="http://localhost:8081/v1",
        ),
        turn_detection=ChineseModel(    # 中文轮次检测
            unlikely_threshold=0.08
        ),
    )
    
    # 设置对话记录保存回调
    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/workspace/logs/transcript_{ctx.room.name}_{current_date}.json"
        with open(filename, 'w+') as f:
            json.dump(session.history.to_dict(), f, indent=4, ensure_ascii=False)
    
    ctx.add_shutdown_callback(write_transcript)
    
    # 启动 RoomIO (房间输入输出)
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()
    
    # 启动会话
    await session.start(
        agent=used_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    
    # 连接到房间
    await ctx.connect()
    
    # 等待第一个参与者
    participant = await ctx.wait_for_participant()
    
    # 注册 RPC 方法 (用于重置对话)
    @ctx.room.local_participant.register_rpc_method("new_conversation")
    async def new_conversation(data: rtc.RpcInvocationData):
        session.interrupt()           # 中断当前回复
        session.clear_user_turn()     # 清除用户轮次
        await session._agent.update_chat_ctx(ChatContext.empty())  # 清空上下文
```

### 执行流程

```
1. Worker 启动
       ↓
2. prewarm() 预加载 VAD 模型
       ↓
3. entrypoint() 被调用
       ↓
4. 创建 Agent 实例 (悠悠/娜娜)
       ↓
5. 创建 AgentSession (配置 STT/LLM/TTS/VAD)
       ↓
6. 注册 shutdown 回调 (保存对话记录)
       ↓
7. 启动 RoomIO (开始接收音频)
       ↓
8. 启动 AgentSession (开始处理对话)
       ↓
9. 等待用户加入房间
       ↓
10. 注册 RPC 方法 (支持外部重置)
```

---

## Agent 基类详解

**文件**: `agents/livekit-agents/livekit/agents/voice/agent.py`

`Agent` 类是所有 AI 代理的**基类**，定义了：
- 核心属性 (指令、工具、聊天上下文)
- 生命周期钩子 (on_enter, on_exit)
- 处理节点 (stt_node, llm_node, tts_node)

### 核心属性

```python
class Agent:
    def __init__(
        self,
        *,
        instructions: str,              # 系统指令 (角色设定)
        chat_ctx: ChatContext = None,   # 聊天上下文
        tools: list = None,             # 工具函数列表
        turn_detection = None,          # 轮次检测配置
        stt = None,                     # STT 组件
        vad = None,                     # VAD 组件
        llm = None,                     # LLM 组件
        tts = None,                     # TTS 组件
        allow_interruptions = None,     # 是否允许打断
        min_endpointing_delay = None,   # 最小端点延迟
        max_endpointing_delay = None,   # 最大端点延迟
    ):
        self._instructions = instructions
        self._tools = tools or []
        self._chat_ctx = chat_ctx or ChatContext.empty()
        self._activity: AgentActivity | None = None  # 活动上下文
```

### 生命周期钩子

```python
async def on_enter(self) -> None:
    """当 Agent 进入会话时调用
    
    典型用法:
    - 生成欢迎回复
    - 初始化状态
    """
    pass

async def on_exit(self) -> None:
    """当 Agent 退出会话时调用
    
    典型用法:
    - 清理资源
    - 保存状态
    """
    pass

async def on_user_turn_completed(
    self, 
    turn_ctx: ChatContext,      # 当前聊天上下文
    new_message: ChatMessage    # 用户新消息
) -> None:
    """用户说完话、LLM 即将回复前调用
    
    典型用法:
    - 修改聊天上下文
    - 编辑用户消息
    - 添加系统提示
    """
    pass
```

### 处理节点 (Pipeline Nodes)

Agent 使用**节点式管道**处理数据流：

```
音频输入 → stt_node → llm_node → tts_node → 音频输出
                ↓              ↓
           转录文本        回复文本
```

#### 1. STT 节点 (语音→文本)

```python
def stt_node(
    self, 
    audio: AsyncIterable[rtc.AudioFrame],  # 音频流
    model_settings: ModelSettings
) -> AsyncIterable[stt.SpeechEvent]:
    """将音频帧转换为语音事件
    
    默认实现:
    1. 检查 STT 是否支持流式
    2. 不支持则用 VAD 包装
    3. 创建流式连接
    4. 转发音频帧
    5. 产出语音事件
    """
    return Agent.default.stt_node(self, audio, model_settings)
```

#### 2. LLM 节点 (文本→回复)

```python
def llm_node(
    self,
    chat_ctx: ChatContext,         # 聊天上下文
    tools: list[FunctionTool],     # 可用工具
    model_settings: ModelSettings
) -> AsyncIterable[llm.ChatChunk]:
    """使用 LLM 生成回复
    
    默认实现:
    1. 获取活动上下文中的 LLM
    2. 创建聊天流
    3. 产出聊天块 (文本 + 工具调用)
    """
    return Agent.default.llm_node(self, chat_ctx, tools, model_settings)
```

#### 3. TTS 节点 (文本→语音)

```python
def tts_node(
    self, 
    text: AsyncIterable[str],      # 文本流
    model_settings: ModelSettings
) -> AsyncIterable[rtc.AudioFrame]:
    """将文本合成为音频
    
    默认实现:
    1. 检查 TTS 是否支持流式
    2. 不支持则用分词器包装
    3. 创建流式连接
    4. 转发文本块
    5. 产出音频帧
    """
    return Agent.default.tts_node(self, text, model_settings)
```

### 动态更新方法

```python
async def update_instructions(self, instructions: str) -> None:
    """动态更新角色指令"""
    if self._activity is None:
        self._instructions = instructions
        return
    await self._activity.update_instructions(instructions)

async def update_tools(self, tools: list) -> None:
    """动态更新工具列表"""
    if self._activity is None:
        self._tools = list(set(tools))
        return
    await self._activity.update_tools(tools)

async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
    """动态更新聊天上下文"""
    if self._activity is None:
        self._chat_ctx = chat_ctx.copy(tools=self._tools)
        return
    await self._activity.update_chat_ctx(chat_ctx)
```

---

## 会话管理

> ⚠️ 注意：`AgentSession` 类在 `voice/agent_session.py` 中定义
> 这里基于代码结构推断其核心逻辑

**文件**: `agents/livekit-agents/livekit/agents/voice/agent_session.py`

`AgentSession` 是**会话的核心管理器**，负责：
- 管理语音识别流
- 协调 LLM 推理
- 控制语音合成流
- 处理 VAD 检测
- 管理轮次检测

### 核心职责

```python
class AgentSession:
    """会话管理器
    
    管理组件:
    - vad: 语音活动检测
    - stt: 语音识别
    - llm: 语言模型
    - tts: 语音合成
    - turn_detection: 轮次检测
    
    关键方法:
    - start(): 启动会话
    - generate_reply(): 生成回复
    - interrupt(): 中断回复
    - clear_user_turn(): 清除用户轮次
    """
```

### 启动流程

```python
async def start(
    self,
    agent: Agent,                      # Agent 实例
    room: rtc.Room,                    # LiveKit 房间
    room_input_options: RoomInputOptions = None,   # 输入配置
    room_output_options: RoomOutputOptions = None, # 输出配置
):
    """启动会话
    
    执行步骤:
    1. 创建 AgentActivity (活动上下文)
    2. 连接房间音频轨道
    3. 启动 RoomIO (输入输出处理器)
    4. 初始化 VAD 流
    5. 启动 STT 流
    6. 启动 LLM 节点
    7. 启动 TTS 节点
    8. 启动轮次检测
    """
    pass
```

### 回复生成

```python
async def generate_reply(self, instructions: str = None):
    """生成回复
    
    执行步骤:
    1. 收集用户输入 (从 STT)
    2. 更新聊天上下文
    3. 调用 LLM 节点生成回复
    4. 通过 TTS 节点合成语音
    5. 发送音频到房间
    6. 发送转录文本
    """
    pass
```

### 中断处理

```python
async def interrupt(self):
    """中断当前回复
    
    使用场景:
    - 用户打断 Agent 说话
    - 新消息到达
    - 外部触发重置
    
    执行步骤:
    1. 停止 TTS 播放
    2. 清空待发送队列
    3. 标记当前回复为已中断
    """
    pass

async def clear_user_turn(self):
    """清除用户轮次
    
    使用场景:
    - 开始新对话
    - 重置上下文
    
    执行步骤:
    1. 清除 STT 缓冲
    2. 重置 VAD 状态
    3. 清空临时转录
    """
    pass
```

---

## 语音活动检测 (pVAD)

**文件**: `agents/fireredchat-plugins/livekit-plugins-fireredchat-pvad/livekit/plugins/fireredchat_pvad/vad.py`

pVAD (Personalized VAD) 是**个性化语音活动检测**，能：
- 检测用户是否在说话
- 区分语音和静音
- 支持说话人嵌入 (声纹识别)

### 核心类结构

```
VAD (主类)
├── VadProcessor (ONNX 推理)
│   ├── pvad.onnx (语音检测模型)
│   └── spkrec-ecapa-voxceleb (声纹提取)
└── VADStream (流式处理)
```

### VAD 加载

```python
@classmethod
def load(
    cls,
    *,
    min_speech_duration: float = 0.16,    # 最小语音时长 (秒)
    min_silence_duration: float = 0.40,   # 最小静音时长 (秒)
    prefix_padding_duration: float = 0.5, # 前缀填充时长
    max_buffered_speech: float = 20.0,    # 最大语音缓冲 (秒)
    activation_threshold: float = 0.5,    # 激活阈值
    sample_rate: int = 16000,             # 采样率
    force_cpu: bool = True,               # 强制 CPU
) -> VAD:
    """加载 VAD 模型
    
    执行步骤:
    1. 创建 VadProcessor (加载 ONNX 模型)
    2. 加载声纹提取器 (SpeechBrain)
    3. 配置 VAD 参数
    4. 返回 VAD 实例
    """
    model = VadProcessor()
    opts = _VADOptions(...)
    return cls(model=model, opts=opts)
```

### VadProcessor 推理

```python
class VadProcessor:
    def __init__(self):
        # 加载 ONNX 模型
        self._session = ort.InferenceSession(
            "resources/pvad.onnx",
            providers=["CPUExecutionProvider"]
        )
        
        # 加载声纹提取器
        self._spk_extractor = SpeakerEmbExtractor(
            "resources/spkrec-ecapa-voxceleb"
        )
        
        # 初始化缓冲
        self.mel_buffer = np.zeros((1, 80, 15))  # Mel 频谱缓冲
        self.gru_buffer = np.zeros((2, 1, 256))  # GRU 状态缓冲
        self.spkemb = np.zeros((1, 192))         # 声纹嵌入
    
    def __call__(self, wav_np):
        """执行推理
        
        输入：160 采样点 (10ms @ 16kHz)
        输出：语音概率 (0.0-1.0)
        """
        # 形状校验
        if wav_np.shape[0] != 160:
            return 0.0
        
        # ONNX 推理
        outputs = self._session.run(None, {
            'input_audio': wav_np,      # 音频输入
            'spkemb': self.spkemb,      # 声纹嵌入
            'mel_buffer': self.mel_buffer,  # Mel 缓冲
            'gru_buffer': self.gru_buffer   # GRU 缓冲
        })
        
        # 提取结果
        raw_prob = outputs[1][0].tolist()[0]  # 语音概率
        self.mel_buffer = outputs[2]          # 更新 Mel 缓冲
        self.gru_buffer = outputs[3]          # 更新 GRU 缓冲
        
        return raw_prob
```

### VADStream 流式处理

```python
async def _main_task(self) -> None:
    """主处理循环
    
    状态机:
    - 静音态 → 语音概率 < 阈值
    - 语音态 → 语音概率 >= 阈值
    
    事件:
    - START_OF_SPEECH: 检测到语音开始
    - END_OF_SPEECH: 检测到语音结束
    - INFERENCE_DONE: 每次推理完成
    """
    
    # 状态变量
    pub_speaking = False           # 是否正在说话
    pub_speech_duration = 0.0      # 语音时长
    pub_silence_duration = 0.0     # 静音时长
    
    async for input_frame in self._input_ch:
        # 1. 收集音频帧
        input_frames.append(input_frame)
        
        # 2. 重采样到 16kHz (如果需要)
        if resampler:
            inference_frames.extend(resampler.push(input_frame))
        
        # 3. 执行推理 (每 10ms)
        while available_samples >= window_size:
            # 转换为 f32
            np.divide(inference_data, max_int16, out=inference_f32)
            
            # 运行推理
            p = await loop.run_in_executor(executor, model, inference_f32)
            
            # 指数平滑
            p = exp_filter.apply(exp=1.0, sample=p)
            
            # 4. 状态转换
            if p >= activation_threshold:
                # 语音态
                speech_threshold_duration += window_duration
                
                if not pub_speaking and speech_threshold_duration >= min_speech_duration:
                    # 静音 → 语音
                    pub_speaking = True
                    send_event(START_OF_SPEECH)
            else:
                # 静音态
                silence_threshold_duration += window_duration
                
                if pub_speaking and silence_threshold_duration >= min_silence_duration:
                    # 语音 → 静音
                    pub_speaking = False
                    send_event(END_OF_SPEECH)
            
            # 5. 发送推理完成事件
            send_event(INFERENCE_DONE)
```

### 声纹更新

```python
def update_speaker(self, frame) -> None:
    """更新说话人声纹
    
    用途：个性化 VAD，只检测特定说话人
    
    执行步骤:
    1. 重采样到 16kHz
    2. 截取 5 秒音频
    3. 提取声纹嵌入 (ECAPA-TDNN)
    4. 归一化
    5. 更新模型声纹
    """
    if not self._speaker_emb_updated:
        # 重采样
        frames = self._ecapa_resampler.push(frame)
        frame = combine_audio_frames(frames)
        
        # 提取声纹
        inference_data = np.array(frame.data[:80000], dtype=np.float32) / 32768.0
        audio = torch.from_numpy(inference_data).unsqueeze(0)
        
        # 运行提取器
        self.spkemb = self._spk_extractor.get_embedding(audio)
```

---

## 轮次检测 (Turn Detector)

**文件**: `agents/fireredchat-plugins/livekit-plugins-fireredchat-turn-detector/livekit/plugins/fireredchat_turn_detector/base.py`

轮次检测器用于**判断用户是否说完话**，基于：
- 聊天上下文分析
- ONNX 模型推理
- 中文/多语言支持

### 核心类结构

```
EOUModelBase (抽象基类)
├── ChineseModel (中文模型)
│   └── _EUORunnerChinese
└── MultilingualModel (多语言模型)
    └── _EUORunnerMultilingual
```

### 模型推理

```python
class _EUORunnerChinese(_InferenceRunner):
    """中文轮次检测推理器"""
    
    def initialize(self) -> None:
        """初始化模型
        
        加载:
        - tokenizer: BERT 分词器
        - ONNX 模型：chinese_best_model_q8.onnx (量化版)
        """
        self.tokenizer_path = "./pretrained_models/tokenizer"
        self.local_path_onnx = "./pretrained_models/chinese_best_model_q8.onnx"
        
        self._session = ort.InferenceSession(
            self.local_path_onnx,
            providers=["CPUExecutionProvider"]
        )
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
            truncation_side="left"
        )
    
    def run(self, data: bytes) -> bytes:
        """执行推理
        
        输入：{"chat_ctx": [{"role": "user", "content": "..."}]}
        输出：{"eou_probability": 0.95, "input": "...", "duration": 0.05}
        """
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx")
        
        # 提取最近用户消息
        text = ""
        for msg in chat_ctx[::-1]:
            if msg["role"] == "user":
                text = msg["content"] + text
            else:
                break
        
        # 移除标点
        text = re.sub("[,.?!]", "", text)
        
        # 分词
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding='max_length',
            add_special_tokens=False,
            return_tensors="np",
            max_length=128
        )
        
        # 推理
        outputs = self._session.run(None, {
            "input_ids": inputs["input_ids"].astype("int64"),
            "attention_mask": inputs["attention_mask"].astype("int64")
        })
        
        # Softmax 获取概率
        eou_probability = softmax(outputs[0]).flatten()[-1]
        
        return json.dumps({
            "eou_probability": float(eou_probability),
            "input": text,
            "duration": round(end_time - start_time, 3)
        })
```

### 预测接口

```python
class ChineseModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float = None):
        super().__init__(unlikely_threshold=unlikely_threshold)
    
    async def predict_end_of_turn(
        self, 
        chat_ctx: ChatContext, 
        *, 
        timeout: float = 1.0  # 1 秒超时
    ) -> float:
        """预测轮次结束概率
        
        执行步骤:
        1. 提取最近用户消息
        2. 序列化为 JSON
        3. 发送到推理执行器
        4. 等待结果 (最多 1 秒)
        5. 返回 EOU 概率
        
        返回值:
        - 0.0-1.0: 轮次结束概率
        - >0.5: 很可能说完了
        - <0.2: 很可能还在说
        """
        messages = []
        for item in chat_ctx.items:
            if item.type != "message":
                continue
            if item.role not in ("user", "assistant"):
                continue
            for cnt in item.content:
                if isinstance(cnt, str):
                    messages.append({"role": item.role, "content": cnt})
                    break
        
        messages = messages[-1:]  # 只取最近一条
        
        json_data = json.dumps({"chat_ctx": messages}).encode()
        
        result = await asyncio.wait_for(
            self._executor.do_inference("lk_end_of_utterance_chinese", json_data),
            timeout=timeout
        )
        
        result_json = json.loads(result.decode())
        return result_json["eou_probability"]
```

### 使用示例

```python
# 创建轮次检测器
turn_detection = ChineseModel(unlikely_threshold=0.08)

# 在 AgentSession 中使用
session = AgentSession(
    ...,
    turn_detection=turn_detection,
)

# 内部逻辑:
# 1. 每次用户消息到达时调用 predict_end_of_turn()
# 2. 如果概率 > 0.5，标记为用户轮次结束
# 3. 触发 LLM 生成回复
```

---

## ASR 集成

**文件**: `agents/fireredchat-plugins/livekit-plugins-firered/livekit/plugins/firered/stt.py`

FireRed ASR 集成封装了**自研语音识别服务**，支持：
- 流式转录
- 实时 WebSocket 连接
- 中英混合识别

### STT 类结构

```python
class STT(stt.STT):
    """FireRed 语音识别
    
    配置选项:
    - model: 模型名称 (FireRedASR-AED-1)
    - base_url: ASR 服务地址
    - api_key: API 密钥
    - language: 语言代码
    - turn_detection: 轮次检测配置
    """
    
    def __init__(
        self,
        *,
        language: str = "en",
        model: str = "gpt-4o-mini-transcribe",
        base_url = NOT_GIVEN,
        api_key = NOT_GIVEN,
        turn_detection = NOT_GIVEN,
        use_realtime: bool = False,  # 使用实时 API
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_realtime,
                interim_results=use_realtime
            )
        )
        
        self._opts = _STTOptions(...)
        self._client = openai.AsyncClient(...)
        self._pool = ConnectionPool(...)  # WebSocket 连接池
```

### WebSocket 连接

```python
async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
    """建立 WebSocket 连接
    
    配置实时转录会话:
    {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "FireRedASR-AED-1",
                "language": "zh",
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 350,
            }
        }
    }
    """
    realtime_config = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": self._opts.model,
                "prompt": self._opts.prompt,
            },
            "turn_detection": self._opts.turn_detection,
        },
    }
    
    if self._opts.language:
        realtime_config["session"]["input_audio_transcription"]["language"] = (
            self._opts.language
        )
    
    # 建立连接
    url = f"{self._client.base_url}/realtime?intent=transcription"
    ws = await session.ws_connect(url, headers={
        "Authorization": f"Bearer {self._client.api_key}",
        "OpenAI-Beta": "realtime=v1",
    })
    
    # 发送配置
    await ws.send_json(realtime_config)
    
    return ws
```

### 流式处理

```python
class SpeechStream(stt.SpeechStream):
    """流式语音识别流"""
    
    async def _run(self) -> None:
        """主处理循环
        
        双任务模式:
        1. send_task: 发送音频到服务器
        2. recv_task: 接收转录结果
        """
        
        async def send_task(ws):
            """发送音频帧
            
            将音频切分为 50ms 块发送
            """
            audio_bstream = AudioByteStream(
                sample_rate=16000,
                samples_per_channel=16000 // 20,  # 50ms
            )
            
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    frames = audio_bstream.write(data.data.tobytes())
                elif isinstance(data, _FlushSentinel):
                    frames = audio_bstream.flush()
                
                for frame in frames:
                    # Base64 编码发送
                    encoded = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame.data).decode(),
                    }
                    await ws.send_json(encoded)
        
        async def recv_task(ws):
            """接收转录结果
            
            处理事件:
            - conversation.item.input_audio_transcription.delta (中间结果)
            - conversation.item.input_audio_transcription.completed (最终结果)
            """
            current_text = ""
            last_interim_at = 0
            
            while True:
                msg = await ws.receive()
                data = json.loads(msg.data)
                msg_type = data.get("type")
                
                if msg_type == "conversation.item.input_audio_transcription.delta":
                    # 中间转录
                    delta = data.get("delta", "")
                    current_text += delta
                    
                    # 每 0.5 秒发送一次
                    if time.time() - last_interim_at > 0.5:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[stt.SpeechData(text=current_text)]
                            )
                        )
                        last_interim_at = time.time()
                
                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    # 最终转录
                    current_text = ""
                    transcript = data.get("transcript", "")
                    
                    if transcript:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[stt.SpeechData(text=transcript)]
                            )
                        )
```

---

## TTS 集成

**文件**: `agents/fireredchat-plugins/livekit-plugins-firered/livekit/plugins/firered/tts.py`

FireRed TTS 集成封装了**自研语音合成服务**，支持：
- 多种语音 ID
- MP3/WAV 格式
- 流式输出

### TTS 类结构

```python
class TTS(tts.TTS):
    """FireRed 语音合成
    
    配置选项:
    - model: 模型名称 (fireredtts1.0)
    - voice: 语音 ID (f531)
    - base_url: TTS 服务地址
    - response_format: 输出格式 (mp3/wav)
    """
    
    def __init__(
        self,
        *,
        model: str = "fireredtts1.0",
        voice: str = "f531",
        base_url = NOT_GIVEN,
        api_key = NOT_GIVEN,
        response_format: str = "mp3",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            response_format=response_format,
        )
        
        self._client = openai.AsyncClient(...)
```

### 语音合成

```python
class ChunkedStream(tts.ChunkedStream):
    """分块语音合成流"""
    
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """执行合成
        
        请求格式:
        POST /v1/audio/speech
        {
            "input": "哈喽，你好呀～",
            "model": "fireredtts1.0",
            "voice": "f531",
            "response_format": "mp3",
            "extra": {"language": "auto", "stream": true}
        }
        """
        oai_stream = self._tts._client.audio.speech.with_streaming_response.create(
            input=self.input_text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format=self._opts.response_format,
            instructions=self._opts.instructions or openai.NOT_GIVEN,
            timeout=httpx.Timeout(10, connect=self._conn_options.timeout),
            extra_body={"language": "auto", "stream": True}
        )
        
        async with oai_stream as stream:
            # 初始化输出
            output_emitter.initialize(
                request_id=stream.request_id,
                sample_rate=24000,
                num_channels=1,
                mime_type=f"audio/{self._opts.response_format}",
            )
            
            # 流式接收音频数据
            async for data in stream.iter_bytes():
                output_emitter.push(data)
            
            # 刷新输出
            output_emitter.flush()
```

---

## 数据流全景

### 完整语音对话流程

```
1. 用户说话
       │
       ▼
2. LiveKit RTC 接收音频轨道
       │
       ▼
3. RoomIO 提取音频帧 (16kHz, 16bit, 单声道)
       │
       ▼
4. pVAD 检测语音活动
       │
       ├─→ 静音：丢弃
       │
       └─→ 语音：继续
              │
              ▼
5. FireRed ASR 流式识别
       │
       ├─→ 中间结果：实时显示
       │
       └─→ 最终结果：发送到 LLM
              │
              ▼
6. 轮次检测器判断是否说完
       │
       ├─→ 未说完：继续收集
       │
       └─→ 说完了：触发回复
              │
              ▼
7. LLM 生成回复文本 (Ollama + Qwen2.5)
       │
       ▼
8. FireRed TTS 合成语音
       │
       ▼
9. LiveKit RTC 发送音频给用户
       │
       ▼
10. 用户听到回复
```

### 状态转换图

```
┌─────────────┐
│   静音态     │
│ (等待语音)   │
└──────┬──────┘
       │ 语音概率 >= 阈值
       │ 持续 >= 0.16s
       ▼
┌─────────────┐
│   语音态     │
│ (收集音频)   │
└──────┬──────┘
       │ 语音概率 < 阈值
       │ 持续 >= 0.40s
       ▼
┌─────────────┐
│  轮次检测    │
│ (判断结束)   │
└──────┬──────┘
       │ EOU 概率 >= 0.5
       ▼
┌─────────────┐
│  LLM 推理    │
│ (生成回复)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  TTS 合成    │
│ (播放回复)   │
└─────────────┘
```

---

## 关键配置参数

### pVAD 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `activation_threshold` | 0.5 | 语音激活阈值 (0-1) |
| `min_speech_duration` | 0.16s | 最小语音时长 |
| `min_silence_duration` | 0.40s | 最小静音时长 |
| `prefix_padding_duration` | 0.5s | 前缀填充时长 |
| `max_buffered_speech` | 20.0s | 最大语音缓冲 |

### 轮次检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `unlikely_threshold` | 0.08 | 不可能结束阈值 |
| `timeout` | 1.0s | 推理超时 |
| `max_history_tokens` | 128 | 最大历史 token 数 |

### ASR 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | FireRedASR-AED-1 | 模型名称 |
| `language` | en | 语言代码 |
| `sample_rate` | 16000 | 采样率 |
| `max_session_duration` | 10min | 会话超时 |

### TTS 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | fireredtts1.0 | 模型名称 |
| `voice` | f531 | 语音 ID |
| `response_format` | mp3 | 输出格式 |
| `sample_rate` | 24000 | 采样率 |

### LLM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | qwen2.5-7b | 模型名称 |
| `base_url` | http://localhost:11434/v1 | Ollama 地址 |
| `temperature` | 0.7 | 温度参数 |

---

## 附录：核心文件索引

| 文件 | 作用 | 代码量 |
|------|------|--------|
| `fireredchat_worker.py` | Worker 入口 | ~150 行 |
| `agent.py` | Agent 基类 | ~800 行 |
| `vad.py` | pVAD 实现 | ~400 行 |
| `base.py` | 轮次检测 | ~200 行 |
| `stt.py` | ASR 集成 | ~350 行 |
| `tts.py` | TTS 集成 | ~150 行 |

---

_代码逻辑详解文档结束_
