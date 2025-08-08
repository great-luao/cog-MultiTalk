# MultiTalk RunPod Docker Environment

这是一个为RunPod优化的MultiTalk运行环境，提供了所有必要的依赖和工具。

## 使用方法

### 1. 构建镜像

```bash
docker build -t multitalk-runtime:latest .
```

### 2. 在RunPod上使用

在RunPod创建Pod时：
- **Container Image**: `multitalk-runtime:latest` (或推送到Docker Hub后的镜像名)
- **GPU Type**: A100 40GB或更高
- **Volume Mount Path**: `/workspace`

### 3. SSH进入容器后使用

```bash
# 进入项目目录
cd /workspace/cog-MultiTalk

# 运行预测
python run_predict.py \
  --image /workspace/inputs/reference.jpg \
  --first-audio /workspace/inputs/audio.wav \
  --output /workspace/outputs/result.mp4
```

## 环境说明

- **Python**: 3.12
- **CUDA**: 12.1
- **包含**: PyTorch, Transformers, Flash Attention等所有必要依赖
- **模型路径**: `/workspace/weights/` (自动下载和管理)

## 参数说明

- `--image`: 参考图像路径
- `--first-audio`: 音频文件路径
- `--second-audio`: 第二个音频（可选，用于多人对话）
- `--prompt`: 场景描述
- `--num-frames`: 生成帧数 (25-201)
- `--sampling-steps`: 采样步数 (2-100)
- `--seed`: 随机种子
- `--turbo`: 启用快速模式
- `--output`: 输出路径