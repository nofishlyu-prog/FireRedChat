# FireRedTTS 模型架构与训练详解

> 📖 文档生成者：小阅 (Xiaoyue)  
> 📅 生成时间：2026-03-10  
> 🎯 模型版本：FireRedTTS-1S

---

## 📑 目录

1. [TTS 概述](#tts-概述)
2. [FireRedTTS-1S 架构](#fireredtts-1s-架构)
3. [模型组件详解](#模型组件详解)
4. [训练方法](#训练方法)
5. [推理流程](#推理流程)
6. [语音克隆技术](#语音克隆技术)
7. [性能优化](#性能优化)
8. [部署配置](#部署配置)
9. [使用限制](#使用限制)

---

## TTS 概述

### FireRedTTS 是什么？

FireRedTTS 是 FireRedTeam 自研的**端到端语音合成模型**，支持：
- 高质量中文语音合成
- 零样本语音克隆 (Zero-shot Voice Cloning)
- 情感控制
- 实时流式合成

### 模型版本

| 版本 | 特点 | 参数量 | 适用场景 |
|------|------|--------|----------|
| FireRedTTS-1 | 基础版 | ~500M | 通用合成 |
| **FireRedTTS-1S** | 单说话人优化 | ~300M | 特定语音克隆 |
| FireRedTTS-M | 多说话人 | ~800M | 多角色对话 |

### 技术栈

```
深度学习框架：PyTorch 2.1+
推理加速：ONNX Runtime / TensorRT
音频编码：Vocoder (HiFi-GAN / BigVGAN)
特征提取：Mel 频谱 (80 维)
文本处理：中文拼音 + 英文 Grapheme
```

### 应用场景

```
1. 语音助手
   └─→ 自然对话回复
   
2. 有声书朗读
   └─→ 长时间文本合成
   
3. 角色配音
   └─→ 多情感、多风格
   
4. 实时对话系统
   └─→ 低延迟流式合成
```

---

## FireRedTTS-1S 架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      文本输入                                │
│            "哈喽，你好呀～"                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   文本前端 (Text Frontend)                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  文本规范化 (Text Normalization)                       │  │
│  │  ├─ 数字转写：123 → "一百二十三"                        │  │
│  │  ├─ 日期转写：2024-01-01 → "二零二四年一月一日"         │  │
│  │  ├─ 多音字消歧：重庆 (chóng qìng)                       │  │
│  │  └─ 英文转写：Hello → "哈喽"                            │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  韵律预测 (Prosody Prediction)                         │  │
│  │  ├─ 分词：哈喽 / ，/ 你 / 好 / 呀 / ～                   │  │
│  │  ├─ 音素转换：ha1 lou5 / ni3 / hao3 / ya5              │  │
│  │  └─ 韵律边界预测                                       │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 音素序列 + 韵律特征
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    编码器 (Encoder)                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Text Encoder (Transformer)                            │  │
│  │  ├─ Embedding: 音素 → 512 维向量                         │  │
│  │  ├─ Transformer Blocks × 4                             │  │
│  │  └─ 输出：(T, 512) 文本编码                             │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Prosody Encoder                                       │  │
│  │  ├─ 韵律嵌入：声调、重音、边界                          │  │
│  │  └─ 输出：(T, 128) 韵律编码                             │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 文本编码 + 韵律编码
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  语音合成解码器 (Decoder)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Variational Autoencoder (VAE)                         │  │
│  │  ├─ 说话人嵌入：(1, 256)                               │  │
│  │  ├─ 情感嵌入：(1, 128)                                 │  │
│  │  └─ 风格令牌：Style Tokens                             │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Decoder (Transformer / Diffusion)                     │  │
│  │  ├─ Cross-Attention (关注文本编码)                      │  │
│  │  ├─ Self-Attention (自回归)                            │  │
│  │  └─ 输出：(T', 80) Mel 频谱序列                         │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Mel 频谱
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    声码器 (Vocoder)                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  HiFi-GAN / BigVGAN                                    │  │
│  │  ├─ 输入：(T', 80) Mel 频谱                             │  │
│  │  ├─ 多尺度感受野卷积                                   │  │
│  │  └─ 输出：(T'×256, 1) 音频波形 (24kHz)                  │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 音频波形
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      音频输出                                │
│                  MP3 / WAV 格式                              │
└─────────────────────────────────────────────────────────────┘
```

### API 接口

**文件**: `agents/fireredchat-plugins/livekit-plugins-firered/livekit/plugins/firered/tts.py`

```python
class TTS(tts.TTS):
    """FireRed TTS 集成"""
    
    def __init__(
        self,
        *,
        model: str = "fireredtts1.0",
        voice: str = "f531",           # 语音 ID
        base_url: str = "http://localhost:8081/v1",
        api_key: str = "notneeded",
        response_format: str = "mp3",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,          # 24kHz 输出
            num_channels=1,             # 单声道
        )
        
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            response_format=response_format,
        )
        
        self._client = openai.AsyncClient(
            base_url=base_url,
            api_key=api_key,
        )
    
    def synthesize(self, text: str) -> ChunkedStream:
        """合成语音"""
        return ChunkedStream(tts=self, input_text=text)


class ChunkedStream(tts.ChunkedStream):
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
            
            # 流式接收音频
            async for data in stream.iter_bytes():
                output_emitter.push(data)
            
            output_emitter.flush()
```

---

## 模型组件详解

### 1. 文本前端

**功能**: 将原始文本转换为音素序列

```python
class TextFrontend:
    def __init__(self):
        # 加载词典
        self.pinyin_dict = load_pinyin_dict()
        self.polyphone_model = load_polyphone_model()  # 多音字模型
    
    def process(self, text):
        # 1. 文本规范化
        text = self.normalize_text(text)
        
        # 2. 分词
        words = self.tokenize(text)
        
        # 3. 音素转换
        phonemes = []
        for word in words:
            if is_chinese(word):
                # 中文：汉字→拼音
                pinyin = self.get_pinyin(word)
                phonemes.extend(pinyin)
            else:
                # 英文：保留或音译
                phonemes.extend(word)
        
        # 4. 韵律预测
        prosody = self.predict_prosody(phonemes)
        
        return phonemes, prosody
    
    def normalize_text(self, text):
        """文本规范化"""
        # 数字转写
        text = re.sub(r'\d+', self.number_to_chinese, text)
        
        # 日期转写
        text = re.sub(r'\d{4}-\d{2}-\d{2}', self.date_to_chinese, text)
        
        # 特殊符号
        text = text.replace('~', '到')
        
        return text
```

**多音字消歧**:
```python
def get_pinyin(self, word):
    """获取拼音 (带多音字消歧)"""
    # 基于上下文预测正确读音
    context = self.get_context(word)
    pinyin = self.polyphone_model.predict(word, context)
    return pinyin
```

### 2. 文本编码器

**架构**: Transformer Encoder

```python
class TextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 1. 音素嵌入
        self.embedding = nn.Embedding(
            num_embeddings=args.vocab_size,  # ~2000 (音素 + 符号)
            embedding_dim=args.embed_dim,     # 512
        )
        
        # 2. Transformer 块
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=args.embed_dim,      # 512
                num_heads=args.num_heads,      # 4
                ffn_dim=args.ffn_dim,          # 2048
                dropout=args.dropout,
            )
            for _ in range(args.num_layers)    # 4
        ])
        
        # 3. 韵律投影
        self.prosody_proj = nn.Linear(args.prosody_dim, args.embed_dim)
    
    def forward(self, phonemes, prosody):
        # phonemes: (B, T)
        x = self.embedding(phonemes)  # (B, T, 512)
        
        # 添加韵律信息
        if prosody is not None:
            x = x + self.prosody_proj(prosody).unsqueeze(1)
        
        # Transformer 编码
        for block in self.blocks:
            x = block(x)
        
        return x  # (B, T, 512)
```

### 3. 说话人嵌入

**功能**: 提取或查找说话人向量

```python
class SpeakerEmbedding:
    def __init__(self, model_dir):
        # 预训练声纹模型
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="spkrec-ecapa-voxceleb",
            savedir=model_dir,
        )
        
        # 预定义语音库
        self.voice_db = load_voice_db(model_dir)
        # f531, f532, m501, m502, ...
    
    def get_speaker_embedding(self, voice_id):
        """获取说话人嵌入
        
        Args:
            voice_id: 语音 ID (如 "f531")
        
        Returns:
            torch.Tensor: (1, 256) 嵌入向量
        """
        if voice_id in self.voice_db:
            return self.voice_db[voice_id]
        else:
            raise ValueError(f"Unknown voice: {voice_id}")
    
    def clone_voice(self, reference_audio):
        """零样本语音克隆
        
        Args:
            reference_audio: 参考音频 (5-10 秒)
        
        Returns:
            torch.Tensor: (1, 256) 克隆的声纹
        """
        # 提取声纹
        embedding = self.speaker_encoder.encode_batch(reference_audio)
        
        # L2 归一化
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
```

### 4. 解码器

**架构**: Transformer Decoder with Variational Inference

```python
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 1. 预网络
        self.prenet = nn.Sequential(
            nn.Linear(args.embed_dim, args.decoder_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        
        # 2. Transformer Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                decoder_dim=args.decoder_dim,   # 512
                num_heads=args.num_heads,       # 4
                ffn_dim=args.ffn_dim,           # 2048
                dropout=args.dropout,
            )
            for _ in range(args.num_decoder_layers)  # 6
        ])
        
        # 3. Mel 投影
        self.mel_proj = nn.Linear(args.decoder_dim, args.n_mel)  # 80
        
        # 4. 停止标记预测
        self.stop_proj = nn.Linear(args.decoder_dim, 1)
    
    def forward(self, text_encoding, speaker_embedding, mel_targets=None):
        """
        Args:
            text_encoding: (B, T, 512) 文本编码
            speaker_embedding: (B, 256) 说话人嵌入
            mel_targets: (B, T', 80) 目标 Mel 频谱 (训练时用)
        
        Returns:
            mel_outputs: (B, T', 80) 预测 Mel 频谱
            stop_outputs: (B, T') 停止概率
        """
        # 1. 初始化解码器输入
        batch_size = text_encoding.size(0)
        max_len = mel_targets.size(1) if mel_targets is not None else 100
        
        # 2. 自回归解码
        mel_outputs = []
        stop_outputs = []
        current_mel = torch.zeros(batch_size, 1, args.n_mel)
        
        for t in range(max_len):
            # 解码器输入
            x = self.prenet(current_mel)
            
            # 添加说话人信息
            x = x + speaker_embedding.unsqueeze(1)
            
            # Decoder 块
            for block in self.decoder_blocks:
                x = block(
                    x,
                    text_encoding,
                    tgt_mask=generate_causal_mask(t+1),
                )
            
            # 预测 Mel
            mel = self.mel_proj(x)  # (B, 1, 80)
            mel_outputs.append(mel)
            
            # 预测停止
            stop = torch.sigmoid(self.stop_proj(x))
            stop_outputs.append(stop)
            
            # 检查是否停止
            if stop > 0.5 and t > 10:
                break
            
            current_mel = mel
        
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        
        return mel_outputs, stop_outputs
```

### 5. 声码器 (Vocoder)

**架构**: HiFi-GAN / BigVGAN

```python
class HiFiGANVocoder(nn.Module):
    """HiFi-GAN 声码器
    
    将 Mel 频谱转换为音频波形
    """
    def __init__(self, args):
        super().__init__()
        
        # 多尺度感受野卷积
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=args.n_mel,
                    out_channels=args.hidden_dim,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                ),
                nn.LeakyReLU(0.1),
            ),
            # ... 多个上采样层
        ])
        
        # 多感受野判别器融合
        self.mrf = MultiReceptiveField(args)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv1d(args.hidden_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
    
    def forward(self, mel):
        """
        Args:
            mel: (B, T, 80) Mel 频谱
        
        Returns:
            audio: (B, T×256) 音频波形 (24kHz)
        """
        # 转置：(B, 80, T)
        mel = mel.transpose(1, 2)
        
        # 上采样
        x = mel
        for layer in self.layers:
            x = layer(x)
        
        # MRF 融合
        x = self.mrf(x)
        
        # 输出
        audio = self.output_layer(x)
        
        return audio.squeeze(1)  # (B, T×256)
```

---

## 训练方法

### 数据集

**训练数据组成**:

| 数据集 | 时长 | 说话人 | 内容 | 用途 |
|--------|------|--------|------|------|
| AISHELL-3 | 85h | 218 人 | 中文朗读 | 基础训练 |
| VCTK | 44h | 110 人 | 英文朗读 | 多语言 |
| 自采数据 | 500h+ | 50 人 | 对话/朗读 | 领域适配 |
| **总计** | **~630h+** | **~380 人** | | |

**数据质量要求**:
- 采样率：24kHz 或 48kHz
- 位深：16bit 或 24bit
- 信噪比：>30dB
- 无背景噪音、无混响

### 数据预处理

```python
class DataPreprocessor:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
    
    def process(self, audio_path, text):
        # 1. 加载音频
        audio, sr = torchaudio.load(audio_path)
        
        # 2. 重采样到 24kHz
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        
        # 3. 音量归一化
        audio = audio / audio.abs().max()
        
        # 4. 修剪静音
        audio = self.trim_silence(audio)
        
        # 5. 提取 Mel 频谱
        mel = self.extract_mel(audio)
        
        # 6. 文本处理
        phonemes, prosody = self.text_frontend.process(text)
        
        return {
            'audio': audio,
            'mel': mel,
            'phonemes': phonemes,
            'prosody': prosody,
        }
    
    def extract_mel(self, audio):
        """提取 Mel 频谱"""
        mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
        )
        
        mel = mel_extractor(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mel.transpose(0, 1)  # (T, 80)
```

### 训练目标

**多任务损失函数**:

```python
class TTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Mel 重建损失 (L1)
        self.mel_loss = nn.L1Loss()
        
        # 2. 停止标记损失 (BCE)
        self.stop_loss = nn.BCELoss()
        
        # 3. 对抗损失 (GAN)
        self.gan_loss = GeneratorAdversarialLoss()
        
        # 4. 特征匹配损失
        self.feat_loss = FeatureMatchingLoss()
        
        # 5. 韵律损失
        self.prosody_loss = ProsodyLoss()
    
    def forward(self, outputs, targets):
        mel_pred, stop_pred = outputs['mel'], outputs['stop']
        mel_tgt, stop_tgt = targets['mel'], targets['stop']
        
        # 1. Mel 损失
        loss_mel = self.mel_loss(mel_pred, mel_tgt)
        
        # 2. 停止损失
        loss_stop = self.stop_loss(stop_pred, stop_tgt)
        
        # 3. GAN 损失 (如果用了判别器)
        loss_gan = self.gan_loss(mel_pred)
        
        # 4. 特征匹配
        loss_feat = self.feat_loss(mel_pred, mel_tgt)
        
        # 5. 韵律损失
        loss_prosody = self.prosody_loss(outputs['prosody'], targets['prosody'])
        
        # 总损失
        total_loss = (
            loss_mel * 1.0 +
            loss_stop * 1.0 +
            loss_gan * 0.1 +
            loss_feat * 1.0 +
            loss_prosody * 0.5
        )
        
        return total_loss, {
            'mel': loss_mel.item(),
            'stop': loss_stop.item(),
            'gan': loss_gan.item(),
        }
```

### 训练配置

```yaml
model:
  type: FireRedTTS-1S
  encoder_dim: 512
  decoder_dim: 512
  speaker_dim: 256
  n_mel: 80
  num_heads: 4
  num_encoder_layers: 4
  num_decoder_layers: 6

training:
  optimizer: AdamW
    lr: 1e-3
    weight_decay: 1e-5
    betas: [0.9, 0.98]
  
  scheduler: NoamLR
    warmup_steps: 4000
    factor: 1.0
  
  batch_size: 32
  max_audio_duration: 10s
  
  epochs: 200
  grad_clip: 1.0
  dropout: 0.1
  
  # 混合精度训练
  use_amp: true

validation:
  metrics:
    - mel_reconstruction_error
    - mos_score  # Mean Opinion Score
    - speaker_similarity
  save_best: mos_score
```

### 训练流程

```
1. 数据加载
   │
   ├─→ 加载音频和文本
   ├─→ 提取 Mel 频谱
   ├─→ 文本→音素转换
   └─→ 动态批处理 (按音频长度)
   
2. 前向传播
   │
   ├─→ 文本编码
   ├─→ 说话人嵌入查找
   ├─→ Mel 解码
   ├─→ 声码器生成音频
   └─→ 计算 Loss
   
3. 反向传播
   │
   ├─→ 梯度计算
   ├─→ 梯度裁剪
   └─→ 优化器更新
   
4. 验证与保存
   │
   ├─→ 每 5000 step 验证
   ├─→ 生成样本音频
   ├─→ 计算 MOS 分数
   └─→ 保存最佳模型
```

### 训练技巧

**1. 教师强制 (Teacher Forcing)**:
```python
# 训练时使用真实 Mel 作为输入
if training:
    decoder_input = mel_targets[:, :-1, :]  # Ground Truth
else:
    decoder_input = predicted_mel  # 自回归
```

**2. 引导注意力 (Guided Attention)**:
```python
# 鼓励对角线注意力模式
def guided_attention_loss(attn, text_lengths, mel_lengths):
    # 生成目标注意力矩阵 (对角线)
    target = generate_diagonal_attention(text_lengths, mel_lengths)
    
    # MSE 损失
    loss = F.mse_loss(attn, target)
    
    return loss
```

**3. 数据增强**:
```python
# 音高扰动
def pitch_shift(mel, semitones):
    return torchaudio.transforms.PitchShift(semitones)(mel)

# 速度扰动
def speed_perturb(mel, speed_factor):
    return F.interpolate(mel, scale_factor=speed_factor)
```

---

## 训练数据构造方法

### 音频录制规范

```python
# 录音质量标准
RECORDING_SPECS = {
    "sample_rate": 24000,      # 24kHz
    "bit_depth": 16,           # 16bit
    "channels": 1,             # 单声道
    "snr_min": 30,             # 信噪比 >30dB
    "room_rt60_max": 0.3,      # 混响时间 <300ms
    "peak_level": -3,          # 峰值电平 -3dBFS
}

# 录音脚本示例
class TTSRecordingScript:
    def __init__(self):
        self.sentences = []
    
    def load_script(self, script_file):
        """加载录音文本"""
        with open(script_file, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f if line.strip()]
        
        # 检查覆盖率
        self.check_coverage()
        
        return self.sentences
    
    def check_coverage(self):
        """检查音素覆盖率"""
        all_phonemes = set()
        
        for sentence in self.sentences:
            phonemes = self.text_to_phonemes(sentence)
            all_phonemes.update(phonemes)
        
        # 检查是否覆盖所有中文音素
        required_phonemes = set(['a', 'o', 'e', 'i', 'u', 'ü', ...])  # 完整音素表
        missing = required_phonemes - all_phonemes
        
        if missing:
            print(f"警告：缺少音素覆盖：{missing}")
            print("建议补充包含这些音素的句子")
        
        return len(missing) == 0
    
    def text_to_phonemes(self, text):
        """文本转音素 (用于覆盖率检查)"""
        from pypinyin import lazy_pinyin
        
        phonemes = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                pinyin = lazy_pinyin(char)[0]
                phonemes.extend(list(pinyin))
            else:
                phonemes.append(char)
        
        return phonemes
    
    def generate_recording_list(self, output_file):
        """生成录音列表"""
        with open(output_file, "w", encoding="utf-8") as f:
            for i, sentence in enumerate(self.sentences):
                f.write(f"{i:04d}\t{sentence}\n")
        
        return output_file
```

### 音频预处理

```python
# 音频预处理管道
class TTSAudioPreprocessor:
    def __init__(self, target_sr=24000):
        self.target_sr = target_sr
    
    def preprocess(self, audio_file, text, output_dir):
        """
        预处理音频
        
        步骤:
        1. 质量检查
        2. 修剪静音
        3. 音量归一化
        4. 重采样
        5. 保存
        """
        # 1. 加载音频
        audio, sr = sf.read(audio_file)
        
        # 2. 质量检查
        quality = self.check_quality(audio, sr)
        if not quality["passed"]:
            print(f"质量检查失败：{quality['issues']}")
            return None
        
        # 3. 修剪静音
        audio = self.trim_silence(audio, sr)
        
        # 4. 音量归一化
        audio = self.normalize_volume(audio, target_db=-3)
        
        # 5. 重采样
        if sr != self.target_sr:
            audio = self.resample(audio, sr, self.target_sr)
        
        # 6. 保存
        output_file = f"{output_dir}/{Path(audio_file).stem}_processed.wav"
        sf.write(output_file, audio, self.target_sr)
        
        # 7. 生成元数据
        metadata = {
            "original_file": str(audio_file),
            "processed_file": output_file,
            "text": text,
            "duration": len(audio) / self.target_sr,
            "sample_rate": self.target_sr,
            "quality": quality
        }
        
        with open(f"{output_file}.meta.json", "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return metadata
    
    def check_quality(self, audio, sr):
        """音频质量检查"""
        issues = []
        passed = True
        
        # 检查峰值
        peak = np.max(np.abs(audio))
        if peak > 0.99:
            issues.append("clipping_detected")
            passed = False
        
        # 检查信噪比
        snr = self.estimate_snr(audio)
        if snr < 30:
            issues.append(f"low_snr:{snr:.1f}dB")
            passed = False
        
        # 检查时长
        duration = len(audio) / sr
        if duration < 0.5:
            issues.append("too_short")
            passed = False
        
        if duration > 15:
            issues.append("too_long")
            passed = False
        
        return {
            "passed": passed,
            "issues": issues,
            "peak": peak,
            "snr": snr,
            "duration": duration
        }
    
    def trim_silence(self, audio, sr, threshold=0.01, min_silence=0.1):
        """修剪首尾静音"""
        # 计算能量
        energy = np.abs(audio)
        
        # 找到语音开始
        start = 0
        for i in range(len(energy)):
            if energy[i] > threshold:
                start = max(0, i - int(min_silence * sr))
                break
        
        # 找到语音结束
        end = len(audio)
        for i in range(len(energy) - 1, -1, -1):
            if energy[i] > threshold:
                end = min(len(audio), i + int(min_silence * sr))
                break
        
        return audio[start:end]
    
    def normalize_volume(self, audio, target_db=-3):
        """音量归一化"""
        # 计算当前 RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        # 计算目标 RMS
        target_rms = 10 ** (target_db / 20)
        
        # 缩放
        scale = target_rms / rms
        normalized = audio * scale
        
        # 限制峰值
        normalized = np.clip(normalized, -0.99, 0.99)
        
        return normalized
    
    def estimate_snr(self, audio):
        """估算信噪比"""
        # 简单实现：假设首尾 100ms 为静音
        noise_start = audio[:1600]  # 100ms @ 16kHz
        noise_power = np.mean(noise_start ** 2)
        
        signal_power = np.mean(audio ** 2)
        
        if noise_power == 0:
            return 999
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
```

### 文本 - 音频对齐检查

```python
# 文本 - 音频对齐验证
class TextAudioAlignChecker:
    def __init__(self, asr_model):
        self.asr_model = asr_model
    
    def check_alignment(self, audio_file, reference_text):
        """
        检查文本和音频是否对齐
        
        方法：用 ASR 识别音频，与参考文本比对
        """
        # ASR 识别
        result = self.asr_model.transcribe([audio_file])
        recognized_text = result[0]["text"]
        
        # 计算编辑距离
        edits = jiwer.compute_measures(reference_text, recognized_text)
        
        # 错误率
        cer = (edits["substitutions"] + edits["deletions"] + edits["insertions"]) / \
              (len(reference_text) + 1e-6)
        
        # 判断
        passed = cer < 0.1  # CER < 10%
        
        return {
            "passed": passed,
            "cer": cer * 100,
            "recognized": recognized_text,
            "reference": reference_text,
            "edits": edits
        }
    
    def batch_check(self, metadata_file):
        """批量检查"""
        results = []
        
        with open(metadata_file, "r") as f:
            for line in f:
                meta = json.loads(line)
                
                result = self.check_alignment(
                    meta["processed_file"],
                    meta["text"]
                )
                
                result["file"] = meta["processed_file"]
                results.append(result)
        
        # 统计
        passed = sum(1 for r in results if r["passed"])
        print(f"总计：{len(results)}, 通过：{passed}, 通过率：{passed/len(results)*100:.1f}%")
        
        # 保存未通过的
        failed = [r for r in results if not r["passed"]]
        with open("alignment_failed.jsonl", "w") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        return results
```

---

## 评测数据构造方法

### 测试集设计

```python
# TTS 测试集构造
class TTSTestSetBuilder:
    def __init__(self):
        self.test_sentences = []
    
    def build_scenarios(self):
        """构建评测场景"""
        scenarios = [
            {
                "name": "neutral",
                "description": "中性陈述句",
                "sentences": [
                    "今天天气不错。",
                    "我现在在北京。",
                    "他今年二十岁。"
                ],
                "min_samples": 20
            },
            {
                "name": "question",
                "description": "疑问句",
                "sentences": [
                    "你在哪里？",
                    "这是你的书吗？",
                    "什么时候出发？"
                ],
                "min_samples": 20
            },
            {
                "name": "exclamation",
                "description": "感叹句",
                "sentences": [
                    "太棒了！",
                    "真好看啊！",
                    "好厉害！"
                ],
                "min_samples": 20
            },
            {
                "name": "numbers",
                "description": "数字",
                "sentences": [
                    "我的电话是 13800138000。",
                    "价格是 99.5 元。",
                    "现在是 2024 年 1 月 15 日。"
                ],
                "min_samples": 20
            },
            {
                "name": "english",
                "description": "中英混合",
                "sentences": [
                    "请用 Python 写一个 Hello World。",
                    "我的邮箱是 test@example.com。",
                    "访问 https://example.com 获取更多信息。"
                ],
                "min_samples": 20
            },
            {
                "name": "long_sentence",
                "description": "长句 (>50 字)",
                "sentences": [
                    "这是一个非常长的句子，用来测试 TTS 系统在处理长文本时的表现，包括气息控制和韵律连贯性。"
                ],
                "min_samples": 10
            }
        ]
        
        return scenarios
    
    def create_test_set(self, output_file):
        """创建测试集"""
        scenarios = self.build_scenarios()
        
        test_set = []
        for scenario in scenarios:
            for sentence in scenario["sentences"]:
                test_set.append({
                    "text": sentence,
                    "scenario": scenario["name"],
                    "description": scenario["description"]
                })
        
        # 保存
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        
        print(f"测试集总计：{len(test_set)} 条")
        
        return test_set
```

### 主观评测 (MOS) 数据准备

```python
# MOS 评测数据准备
class MOSDataPreparation:
    def __init__(self):
        self.samples = []
    
    def prepare_mos_test(self, tts_outputs, output_dir):
        """
        准备 MOS 主观评测
        
        Args:
            tts_outputs: TTS 生成的音频列表
            output_dir: 输出目录
        """
        # 创建 HTML 评测界面
        self.create_html_interface(tts_outputs, output_dir)
        
        # 创建评分表
        self.create_rating_sheet(tts_outputs, output_dir)
    
    def create_html_interface(self, tts_outputs, output_dir):
        """创建 HTML 评测界面"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>TTS MOS 评测</title>
    <style>
        .sample { margin: 20px 0; padding: 15px; border: 1px solid #ccc; }
        .rating { margin: 10px 0; }
        .comment { width: 100%; height: 60px; }
    </style>
</head>
<body>
    <h1>TTS 主观评测 (MOS)</h1>
    <p>请对每个样本的语音质量进行评分 (1-5 分)</p>
    <ul>
        <li>5 分：完美，无瑕疵</li>
        <li>4 分：良好，有小问题</li>
        <li>3 分：一般，可接受</li>
        <li>2 分：较差，影响理解</li>
        <li>1 分：很差，无法理解</li>
    </ul>
"""
        
        for i, sample in enumerate(tts_outputs):
            html += f"""
    <div class="sample">
        <h3>样本 {i+1}</h3>
        <p>文本：{sample['text']}</p>
        <audio controls>
            <source src="{sample['audio_file']}" type="audio/wav">
        </audio>
        <div class="rating">
            <label>评分：</label>
            <select name="sample_{i}_rating">
                <option value="5">5 分</option>
                <option value="4">4 分</option>
                <option value="3">3 分</option>
                <option value="2">2 分</option>
                <option value="1">1 分</option>
            </select>
        </div>
        <textarea class="comment" name="sample_{i}_comment" placeholder="备注 (可选问题描述)..."></textarea>
    </div>
"""
        
        html += """
    <button onclick="submit()">提交评分</button>
    <script>
        function submit() {
            // 收集评分并提交
            alert("评分已提交");
        }
    </script>
</body>
</html>
"""
        
        with open(f"{output_dir}/mos_test.html", "w", encoding="utf-8") as f:
            f.write(html)
    
    def create_rating_sheet(self, tts_outputs, output_dir):
        """创建 Excel 评分表"""
        import pandas as pd
        
        data = {
            "样本 ID": range(1, len(tts_outputs) + 1),
            "文本": [s["text"] for s in tts_outputs],
            "场景": [s["scenario"] for s in tts_outputs],
            "MOS 评分": [""] * len(tts_outputs),
            "问题类型": [""] * len(tts_outputs),
            "备注": [""] * len(tts_outputs)
        }
        
        df = pd.DataFrame(data)
        df.to_excel(f"{output_dir}/mos_rating_sheet.xlsx", index=False)
```

---

## 评测方法

### 客观指标

```python
# TTS 客观评测指标
class TTSMetrics:
    def __init__(self):
        pass
    
    def calculate_all(self, reference_audio, synthesized_audio):
        """
        计算所有客观指标
        
        Args:
            reference_audio: 真实录音 (如有)
            synthesized_audio: TTS 合成音频
        
        Returns:
            dict: 各项指标
        """
        metrics = {}
        
        # 1. 声学特征相似度
        metrics.update(self.calculate_acoustic_similarity(
            reference_audio, synthesized_audio
        ))
        
        # 2. 韵律相似度
        metrics.update(self.calculate_prosody_similarity(
            reference_audio, synthesized_audio
        ))
        
        # 3.  intelligibility (可懂度)
        metrics.update(self.calculate_intelligibility(
            synthesized_audio
        ))
        
        return metrics
    
    def calculate_acoustic_similarity(self, ref_audio, syn_audio):
        """声学特征相似度"""
        # 提取 Mel 频谱
        ref_mel = self.extract_mel(ref_audio)
        syn_mel = self.extract_mel(syn_audio)
        
        # 调整长度
        min_len = min(len(ref_mel), len(syn_mel))
        ref_mel = ref_mel[:min_len]
        syn_mel = syn_mel[:min_len]
        
        # 计算 MSE
        mel_mse = np.mean((ref_mel - syn_mel) ** 2)
        
        # 计算余弦相似度
        mel_cosine = np.dot(ref_mel.flatten(), syn_mel.flatten()) / \
                     (np.linalg.norm(ref_mel.flatten()) * np.linalg.norm(syn_mel.flatten()))
        
        return {
            "mel_mse": mel_mse,
            "mel_cosine_similarity": mel_cosine
        }
    
    def calculate_prosody_similarity(self, ref_audio, syn_audio):
        """韵律相似度 (F0, 时长)"""
        # 提取基频 (F0)
        ref_f0 = self.extract_f0(ref_audio)
        syn_f0 = self.extract_f0(syn_audio)
        
        # F0 相关系数
        min_len = min(len(ref_f0), len(syn_f0))
        f0_corr = np.corrcoef(ref_f0[:min_len], syn_f0[:min_len])[0, 1]
        
        # F0 RMSE
        f0_rmse = np.sqrt(np.mean((ref_f0[:min_len] - syn_f0[:min_len]) ** 2))
        
        return {
            "f0_correlation": f0_corr if not np.isnan(f0_corr) else 0,
            "f0_rmse": f0_rmse
        }
    
    def calculate_intelligibility(self, syn_audio):
        """
        可懂度 (使用 ASR 识别率间接评估)
        
        需要先有参考文本
        """
        # 这个方法需要 ASR 服务
        # 这里只是示例
        return {
            "intelligibility_note": "需要使用 ASR 识别率评估"
        }
    
    def extract_mel(self, audio, sr=24000):
        """提取 Mel 频谱"""
        mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            n_mels=80,
        )
        
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        mel = mel_extractor(audio_tensor)
        
        return torch.log(torch.clamp(mel, min=1e-5)).squeeze(0).transpose(0, 1).numpy()
    
    def extract_f0(self, audio, sr=24000):
        """提取基频 (F0)"""
        import librosa
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # 填充未定义值
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return f0

# 使用示例
metrics = TTSMetrics()
results = metrics.calculate_all(
    reference_audio="reference.wav",
    synthesized_audio="synthesized.wav"
)
print(f"Mel 相似度：{results['mel_cosine_similarity']:.4f}")
```

### 主观评测 (MOS) 分析

```python
# MOS 数据分析
class MOSAnalyzer:
    def __init__(self):
        self.ratings = []
    
    def load_ratings(self, ratings_file):
        """加载评分数据"""
        import pandas as pd
        
        df = pd.read_excel(ratings_file)
        self.ratings = df.to_dict("records")
        
        return self.ratings
    
    def calculate_mos(self):
        """计算平均 MOS 分数"""
        ratings = [r["MOS 评分"] for r in self.ratings if r["MOS 评分"]]
        
        if not ratings:
            return None
        
        mos = np.mean(ratings)
        mos_std = np.std(ratings)
        mos_ci = 1.96 * mos_std / np.sqrt(len(ratings))  # 95% 置信区间
        
        return {
            "mos": mos,
            "std": mos_std,
            "ci_95": mos_ci,
            "count": len(ratings)
        }
    
    def analyze_by_scenario(self):
        """分场景分析"""
        scenarios = {}
        
        for r in self.ratings:
            scenario = r.get("场景", "unknown")
            if scenario not in scenarios:
                scenarios[scenario] = []
            
            if r["MOS 评分"]:
                scenarios[scenario].append(r["MOS 评分"])
        
        results = {}
        for scenario, ratings in scenarios.items():
            results[scenario] = {
                "mos": np.mean(ratings),
                "std": np.std(ratings),
                "count": len(ratings)
            }
        
        return results
    
    def analyze_problem_types(self):
        """分析问题类型"""
        problem_types = {}
        
        for r in self.ratings:
            problem = r.get("问题类型", "")
            if problem:
                problem_types[problem] = problem_types.get(problem, 0) + 1
        
        return problem_types
    
    def generate_report(self, output_file):
        """生成评测报告"""
        mos_stats = self.calculate_mos()
        scenario_stats = self.analyze_by_scenario()
        problem_stats = self.analyze_problem_types()
        
        report = f"""# TTS MOS 评测报告

## 总体表现

- **平均 MOS**: {mos_stats['mos']:.2f} ± {mos_stats['ci_95']:.2f} (95% CI)
- **标准差**: {mos_stats['std']:.2f}
- **样本数**: {mos_stats['count']}

## 分场景表现

| 场景 | MOS | 标准差 | 样本数 |
|------|-----|--------|--------|
"""
        
        for scenario, stats in scenario_stats.items():
            report += f"| {scenario} | {stats['mos']:.2f} | {stats['std']:.2f} | {stats['count']} |\n"
        
        report += f"""
## 主要问题

"""
        
        for problem, count in sorted(problem_stats.items(), key=lambda x: -x[1]):
            report += f"- {problem}: {count} 次\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
```

---

## 推理流程

### 完整推理链路

```
1. 文本输入
   │
   ▼
2. 文本前端处理
   │
   ├─→ 文本规范化
   ├─→ 分词
   ├─→ 音素转换
   └─→ 韵律预测
   │
   ▼
3. 文本编码
   │
   └─→ Transformer Encoder
   │
   ▼
4. 说话人嵌入查找
   │
   └─→ voice_id → (1, 256) 向量
   │
   ▼
5. Mel 解码 (自回归)
   │
   ├─→ 初始化：mel_0 = zeros
   ├─→ 循环直到停止标记
   │   ├─→ Decoder 前向
   │   ├─→ 预测下一帧 Mel
   │   └─→ 检查停止概率
   │
   └─→ 输出：(T', 80) Mel 序列
   │
   ▼
6. 声码器合成
   │
   └─→ HiFi-GAN: Mel → 音频波形
   │
   ▼
7. 音频编码
   │
   ├─→ 编码为 MP3/WAV
   └─→ 返回给用户
```

### 自回归解码

```python
@torch.no_grad()
def synthesize(self, text, voice_id):
    """合成语音"""
    # 1. 文本处理
    phonemes, prosody = self.text_frontend.process(text)
    phoneme_ids = self.phoneme_to_ids(phonemes)
    
    # 2. 编码
    text_encoding = self.encoder(phoneme_ids.unsqueeze(0), prosody)
    
    # 3. 说话人嵌入
    speaker_emb = self.speaker_db.get(voice_id)
    
    # 4. 自回归解码
    mel_outputs = []
    stop_outputs = []
    current_mel = torch.zeros(1, 1, 80)
    
    max_steps = 1000  # 防止无限循环
    for t in range(max_steps):
        # Decoder 前向
        mel_pred, stop_pred = self.decoder.step(
            text_encoding,
            speaker_emb,
            current_mel,
        )
        
        mel_outputs.append(mel_pred)
        stop_outputs.append(stop_pred)
        
        # 检查停止
        if stop_pred.sigmoid() > 0.5 and t > 20:
            break
        
        current_mel = mel_pred
    
    # 5. 拼接输出
    mel_outputs = torch.cat(mel_outputs, dim=1)  # (1, T', 80)
    
    # 6. 声码器
    audio = self.vocoder(mel_outputs)  # (1, T'×256)
    
    return audio.squeeze(0).cpu().numpy()
```

### 流式合成

```python
async def synthesize_streaming(self, text, voice_id):
    """流式合成 (边生成边发送)"""
    
    # 分句处理
    sentences = split_sentences(text)
    
    for sentence in sentences:
        # 合成当前句
        audio_chunk = self.synthesize(sentence, voice_id)
        
        # 立即发送
        yield audio_chunk
        
        # 句间短暂停顿
        yield generate_silence(0.1)  # 100ms 静音
```

---

## 语音克隆技术

### 零样本语音克隆

**原理**: 从参考音频提取声纹，用于合成新语音

```python
class VoiceCloner:
    def clone(self, reference_audio, target_text):
        """
        Args:
            reference_audio: 参考音频 (5-10 秒)
            target_text: 要合成的文本
        
        Returns:
            audio: 合成的语音
        """
        # 1. 提取声纹
        speaker_emb = self.extract_speaker_embedding(reference_audio)
        
        # 2. 文本处理
        phonemes, prosody = self.text_frontend.process(target_text)
        
        # 3. 合成
        audio = self.tts.synthesize(
            phonemes=phonemes,
            speaker_embedding=speaker_emb,
        )
        
        return audio
    
    def extract_speaker_embedding(self, audio):
        """提取声纹嵌入"""
        # 重采样到 16kHz
        audio = resample(audio, 16000)
        
        # 截取 5 秒
        audio = audio[:16000*5]
        
        # 归一化
        audio = audio / audio.abs().max()
        
        # 提取嵌入
        embedding = self.speaker_encoder(audio.unsqueeze(0))
        
        # L2 归一化
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding  # (1, 256)
```

### 预定义语音库

**FireRedTTS 语音 ID**:

| ID | 性别 | 风格 | 适用场景 |
|----|------|------|----------|
| f531 | 女 | 温柔、亲切 | 通用助手 |
| f532 | 女 | 活泼、可爱 | 聊天机器人 |
| f533 | 女 | 专业、正式 | 新闻播报 |
| m501 | 男 | 沉稳、可靠 | 商务助手 |
| m502 | 男 | 阳光、开朗 | 娱乐应用 |
| m503 | 男 | 磁性、深情 | 有声书 |

**注意**: 根据 FireRedTTS 许可证，语音 f531 等仅限**非商业用途**使用。

---

## 性能优化

### 1. 模型量化

**INT8 量化**:
```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv1d},
    dtype=torch.qint8,
)

# 量化后效果:
# - 模型大小：减少 75%
# - 推理速度：提升 2-3x
# - 质量损失：<1% MOS
```

### 2. ONNX 导出

```python
# 导出为 ONNX
torch.onnx.export(
    model,
    (phoneme_ids, speaker_emb),
    "fireredtts.onnx",
    input_names=['phonemes', 'speaker'],
    output_names=['mel', 'stop'],
    dynamic_axes={
        'phonemes': {1: 'text_length'},
        'mel': {1: 'mel_length'},
    },
    opset_version=14,
)

# ONNX Runtime 推理
session = onnxruntime.InferenceSession("fireredtts.onnx")
mel, stop = session.run(None, {
    'phonemes': phoneme_ids.numpy(),
    'speaker': speaker_emb.numpy(),
})
```

### 3. 批处理优化

```python
# 多文本批处理
def batch_synthesize(texts, voice_id):
    # 按长度分组
    batches = bucket_by_length(texts)
    
    results = []
    for batch in batches:
        # 批处理编码
        phonemes_batch = [frontend.process(t) for t in batch]
        
        # 批处理解码
        mel_batch = decoder.batch_forward(phonemes_batch)
        
        # 批处理声码器
        audio_batch = vocoder.batch_synthesize(mel_batch)
        
        results.extend(audio_batch)
    
    return results
```

### 4. 缓存策略

```python
@lru_cache(maxsize=100)
def get_speaker_embedding_cached(voice_id):
    return speaker_db.get(voice_id)

@lru_cache(maxsize=1000)
def text_to_phonemes_cached(text):
    return frontend.process(text)
```

### 5. 性能基准

| 配置 | RTF | 延迟 | MOS |
|------|-----|------|-----|
| GPU (RTX 3080) | 0.02 | 100ms | 4.5 |
| GPU (T4) | 0.05 | 200ms | 4.5 |
| CPU (8 核) | 0.3 | 800ms | 4.5 |

**RTF**: 0.02 表示合成 1 秒音频需要 0.02 秒 (50 倍实时)

**MOS**: Mean Opinion Score (1-5 分，5 为最佳)

---

## 部署配置

### Docker 部署

**运行服务**:
```bash
# 下载模型 (注意：仅限非商业用途)
huggingface-cli download --resume-download --repo-type model \
  FireRedTeam/FireRedTTS-1S --revision fireredtts1s_4_chat \
  --local-dir ./tts_4_chat

# 运行容器
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

### API 配置

**svc.yaml**:
```yaml
model:
  path: /workspace/models/redtts
  device: cuda  # 或 cpu
  
server:
  port: 8081
  workers: 4
  timeout: 30
  
cache:
  enabled: true
  max_size: 1000
  
logging:
  level: INFO
  format: json
```

### API 使用示例

**cURL**:
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

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8081/v1/audio/speech",
    json={
        "input": "哈喽，你好呀～",
        "voice": "f531",
        "response_format": "mp3"
    }
)

with open("audio.mp3", "wb") as f:
    f.write(response.content)
```

---

## 使用限制

### 许可证限制

**FireRedTTS-1S 模型权重** (`FireRedTeam/FireRedTTS-1S`, revision `fireredtts1s_4_chat`):

⚠️ **仅限非商业用途使用**

**限制条款**:
1. ❌ 不得用于商业产品或服务
2. ❌ 不得使用语音 f531 及其衍生声音训练/蒸馏自己的模型
3. ✅ 可用于个人项目、研究、教育
4. ✅ 可用于开源项目 (需遵守相同许可证)

**合规建议**:
- 商业用途：联系 FireRedTeam 获取商业许可证
- 自研模型：使用 Apache 2.0 许可的开源 TTS (如 VITS, FastSpeech2)

### 技术限制

| 限制 | 说明 |
|------|------|
| 最大文本长度 | 500 字符/次 |
| 支持语言 | 中文 (主要)、英文 (有限) |
| 情感控制 | 有限 (通过文本表达) |
| 实时性 | 流式合成有 100-200ms 延迟 |

---

## 参考资料

- **HuggingFace**: https://huggingface.co/FireRedTeam/FireRedTTS-1S
- **HiFi-GAN 论文**: https://arxiv.org/abs/2010.05646
- **VITS 论文**: https://arxiv.org/abs/2106.06103
- **FireRedChat Demo**: https://fireredteam.github.io/demos/firered_chat/

---

_FireRedTTS 模型架构文档结束_
