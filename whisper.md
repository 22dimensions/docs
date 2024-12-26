## 两个项目介绍

- [m-bain/whisperX: WhisperX: Automatic Speech Recognition with Word-level Timestamps (& Diarization)](https://github.com/m-bain/whisperX) 
- [shashikg/WhisperS2T: An Optimized Speech-to-Text Pipeline for the Whisper Model Supporting Multiple Inference Engine](https://github.com/shashikg/WhisperS2T)

他们功能一致，都是ASR语音识别框架，区别在于：
- WhisperS2T（**claimed 2.3X speed improvement over [WhisperX](https://github.com/m-bain/whisperX/tree/main)**）相较于 Whisper性能大幅提升，但产品线验证没有收益
- WhisperX 使用 [SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2](https://github.com/SYSTRAN/faster-whisper)作为whisper模型的**运行后端**
- WhisperS2T 支持whisper模型的多个运行后端（CTranslate2、HuggingFace Model with FlashAttention2、Original OpenAI Model）

## 产品线 MindIE-TorchModelZoo适配方案

### whisperX

适配代码 [MindIE/MindIE-Torch/built-in/audio/mindie_whisperx/readme.md · Ascend/ModelZoo-PyTorch - 码云 - 开源中国](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/audio/mindie_whisperx/readme.md)

- 使用**mindtorch部署高性能版本的whisper-large-v3模型的 执行后端**。将开源的whisperX中的**语音切分和自动组batch**的能力迁移过来，达到提升性能的目的
- 通过 patch 定制修改了 VAD(语音活动检测) 和 Whisper Larger V3模型的源代码，然后使用mindietorch编译执行模型，自定义了MindIEPipeline串联执行流程

## 社区适配方案
- mindie方案需要定制修改模型源代码，无法合入社区
- 另外mindie社区版安装包不能公开获取，需要登陆华为账号
	- [简介-MindIE Torch开发指南-PyTorch编译优化-MindIE1.0.RC3开发文档-昇腾社区](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindietorch/Torchdev/mindie_torch0001.html)


whisperX 涉及四个模型
- vad 语音活动检测
- whisper 语音转文本
- Diarization 语音分割
- alignment 对齐

依赖的项目
- [pyannote audio](https://github.com/pyannote/pyannote-audio) (vad & Diarization)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2) (whisper)
- transformers (alignment)

适配方案
- 引入 **torch_npu**
- 当运行设备为NPU时，选择 transformers/pytorch 后端执行，不使用faster_whisper后端 
	- [openai/whisper-large-v3 · Hugging Face](https://huggingface.co/openai/whisper-large-v3)

用户使用NPU推理的示例代码
```python
import whisperx
import gc 

##### device = "cuda" 
###################################################################
device = "npu:0"
##################################################################

audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
```
