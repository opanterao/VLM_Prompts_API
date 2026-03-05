# VLMVideoPrompts

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Supported-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**ComfyUI 节点：使用 VLM 模型反推图像提示词并生成视频提示词**

</div>

## 简介

VLMVideoPrompts 是一个 ComfyUI 自定义节点库，用于：
- 🎨 使用 VLM（视觉语言模型）反推图像提示词
- 🎬 根据多张图片生成连续性视频提示词
- ✨ 增强和优化视频脚本

## 特性

- 支持调用本地 vLLM 部署的 VLM 模型
- 支持多种 VLM 模型（Qwen2-VL、LLaVA 等）
- 灵活的配置选项（API 地址、模型名称、温度等）
- 自动处理 ComfyUI 图像格式
- 错误处理和诊断信息

## 效果演示

### 输入
1. 青铜门环特写
2. 木门全景
3. 无人机视角

### 输出（分镜头模式）
```
镜头1: 精美的青铜门环，细节清晰，古典风格，运镜：固定特写
镜头2: 完整的木门框架，老旧的木质纹理，运镜：缓慢拉远
镜头3: 乡村庭院全景，石柱和天空，运镜：上升鸟瞰
```

### 输出（连续镜头模式）
画面开始于：一个精美的青铜门环特写，细节清晰可见，古典风格浓郁。然后镜头缓慢拉远，显示出完整的木门框架，木质纹理清晰可见。随后无人机视角上升，展现出整个乡村庭院的全貌，远处是覆盖着苔藓的石柱和瓦屋顶。最后画面淡出，暗示时间的流逝。

## 安装方法

### 方法1：复制到 ComfyUI 节点目录

```bash
# Windows
xcopy /E /I "E:\AI_Tools\comfyui_nodes\VLMVideoPrompts" "C:\path\to\ComfyUI\custom_nodes\VLMVideoPrompts"

# Linux/Mac
cp -r /path/to/VLMVideoPrompts /path/to/ComfyUI/custom_nodes/
```

### 方法2：克隆仓库

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/opanterao/VLM_Prompts_API.git VLMVideoPrompts
```

### 方法3：创建符号链接（推荐开发版）

```bash
# Linux/Mac
cd /path/to/ComfyUI/custom_nodes
ln -s /path/to/VLMVideoPrompts VLMVideoPrompts

# Windows (管理员权限)
mklink /D "C:\path\to\ComfyUI\custom_nodes\VLMVideoPrompts" "E:\AI_Tools\comfyui_nodes\VLMVideoPrompts"
```

## 节点说明

### 1. VLM图像转视频提示词 (VLMImageToVideoPrompt)

**功能**：输入1-6张图片，反推每张图片的提示词，并生成视频脚本

**输入**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| api_url | STRING | http://localhost:8000 | vLLM API 地址 |
| model_name | STRING | Qwen2-VL-72B-Instruct | 模型名称 |
| api_key | STRING | (空) | API Key（本地部署留空） |
| temperature | FLOAT | 0.7 | 采样温度 (0.0-2.0) |
| max_tokens | INT | 2048 | 最大生成 token 数 |
| image1-image6 | IMAGE | (可选) | 输入图片（最多6张） |

**输出**：

| 参数 | 类型 | 说明 |
|------|------|------|
| image_prompts | STRING | 每张图片的反推提示词 |
| video_prompts_split | STRING | 分镜头模式提示词 |
| video_prompts_continuous | STRING | 连续镜头模式提示词 |

---

### 2. VLM单图提示词反推 (VLMSingleImagePrompt)

**功能**：输入单张图片，使用自定义提示词反推图像内容

**输入**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| api_url | STRING | http://localhost:8000 | vLLM API 地址 |
| model_name | STRING | Qwen2-VL-72B-Instruct | 模型名称 |
| api_key | STRING | (空) | API Key |
| prompt | STRING | 请详细描述这张图片的所有细节 | 自定义提示词 |
| temperature | FLOAT | 0.7 | 采样温度 |
| max_tokens | INT | 2048 | 最大 token 数 |
| image | IMAGE | (可选) | 输入图片 |

**输出**：
- STRING: 反推的提示词

---

### 3. 视频提示词增强 (VideoPromptEnhancer)

**功能**：对已有的视频提示词进行增强优化

**增强类型**：

| 类型 | 说明 |
|------|------|
| 分镜头模式 | 整理成标准的分镜头脚本格式 |
| 连续镜头模式 | 整理成连贯的视频脚本 |
| 运镜增强 | 添加专业摄影运镜术语 |
| 风格统一 | 统一视觉风格和语言风格 |

---

## 使用示例

### 1. 启动 vLLM 服务

```bash
# 使用 Qwen2-VL 模型
vllm serve Qwen/Qwen2-VL-72B-Instruct --dtype half --host 0.0.0.0 --port 8000

# 或使用其他 VLM 模型
vllm serve llava-hf/llava-1.5-7b-hf --dtype half
```

### 2. 在 ComfyUI 中使用

1. 重启 ComfyUI 使节点生效
2. 在节点列表中找到 `VLMVideo` 分类
3. 拖入 `VLM图像转视频提示词` 节点
4. 连接图像加载节点（如 Load Image）
5. 配置 API 参数（URL、模型名称）
6. 运行工作流获取提示词

### 3. 典型工作流

```
[Load Image] → [Load Image] → [VLMImageToVideoPrompt] → [Save Text] / [Connect to T2V Node]
                [Load Image] ↗
```

## 支持的模型

本节点支持所有兼容 OpenAI API 的 VLM 模型，包括：

- **Qwen2-VL** 系列
  - Qwen2-VL-2B-Instruct
  - Qwen2-VL-7B-Instruct
  - Qwen2-VL-72B-Instruct

- **LLaVA** 系列
  - llava-v1.5-7b-hf
  - llava-v1.5-13b-hf

- **其他兼容模型**
  - bakllava
  - Yi-VL

## 依赖

确保已安装以下 Python 包：

```bash
pip install requests Pillow torch
```

## 常见问题

### Q: 图像转换失败怎么办？

A: 检查图像格式，确保使用 ComfyUI 标准图像节点输出。如果问题持续，请查看错误信息中的 shape 和 dtype。

### Q: API 连接失败怎么办？

A: 
1. 确认 vLLM 服务已启动
2. 检查 API 地址和端口是否正确
3. 确认防火墙允许访问

### Q: 生成质量不好怎么办？

A: 
1. 尝试降低 temperature（0.3-0.5）
2. 增加 max_tokens
3. 使用更大的 VLM 模型

### Q: 如何使用外部 API？

A: 在 api_key 参数中填写 API Key，支持 OpenAI 兼容的 API 服务。

## 项目结构

```
VLMVideoPrompts/
├── __init__.py      # 节点初始化
├── nodes.py         # 节点核心代码
└── README.md        # 使用说明
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 更新日志

### v1.0.0 (2026-03-06)
- 初始版本
- 支持 3 个核心节点
- 支持 VLM 图像反推和视频提示词生成

---

<div align="center">

**如果对你有帮助，欢迎 Star ⭐**

</div>