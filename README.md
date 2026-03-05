# VLMVideoPrompts

ComfyUI 节点：使用 VLM 模型反推图像提示词并生成视频提示词

## 安装方法

### 方法1：复制到 ComfyUI 节点目录

```bash
# 复制整个 VLMVideoPrompts 文件夹到 ComfyUI 的 custom_nodes 目录
cp -r E:\AI_Tools\comfyui_nodes\VLMVideoPrompts /path/to/ComfyUI/custom_nodes/
```

### 方法2：创建符号链接

```bash
# 在 ComfyUI 的 custom_nodes 目录下创建符号链接
cd /path/to/ComfyUI/custom_nodes
ln -s E:\AI_Tools\comfyui_nodes\VLMVideoPrompts VLMVideoPrompts
```

## 节点说明

### 1. VLM图像转视频提示词 (VLMImageToVideoPrompt)

**功能**：输入多张图片，反推每张图片的提示词，并生成视频脚本

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| api_url | STRING | vLLM API 地址（如 http://localhost:8000） |
| model_name | STRING | 模型名称（如 Qwen2-VL-72B-Instruct） |
| api_key | STRING | API Key（本地部署留空） |
| temperature | FLOAT | 采样温度 (0.0-2.0) |
| max_tokens | INT | 最大生成 token 数 |
| image1-image6 | IMAGE | 输入图片（可选1-6张） |

**输出**：
| 参数 | 类型 | 说明 |
|------|------|------|
| image_prompts | STRING | 每张图片的反推提示词 |
| video_prompts_split | STRING | 分镜头模式提示词 |
| video_prompts_continuous | STRING | 连续镜头模式提示词 |

### 2. VLM单图提示词反推 (VLMSingleImagePrompt)

**功能**：输入单张图片，反推提示词

**输入**：
- api_url: API 地址
- model_name: 模型名称
- api_key: API Key
- prompt: 自定义提示词
- image: 输入图片

**输出**：
- STRING: 反推的提示词

### 3. 视频提示词增强 (VideoPromptEnhancer)

**功能**：对已有的视频提示词进行增强优化

**增强类型**：
- 分镜头模式：整理成分镜头脚本
- 连续镜头模式：整理成连贯脚本
- 运镜增强：添加专业运镜术语
- 风格统一：统一视觉风格

## 使用示例

1. 确保 vLLM 服务已启动：
```bash
vllm serve Qwen/Qwen2-VL-72B-Instruct --dtype half
```

2. 在 ComfyUI 中加载节点

3. 连接图像输入，配置 API 参数

4. 运行工作流获取提示词

## 依赖

- requests
- Pillow
- torch
- torchvision
- transformers (用于 vLLM)

## 注意事项

- 确保 vLLM 服务正在运行
- 根据显存情况选择合适的模型
- API Key 在本地部署时留空