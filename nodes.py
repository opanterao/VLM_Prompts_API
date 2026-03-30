import base64
import requests
from io import BytesIO
from PIL import Image
import torch


def validate_url(url):
    """Validate API URL format"""
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def pil2base64(img):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def tensor_to_pil(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    try:
        # 确保是 4D tensor，取第一张图
        if tensor.dim() == 4:
            tensor = tensor[0]  # (H, W, C)

        # 获取形状信息用于调试
        h, w = tensor.shape[:2]
        channels = tensor.shape[2] if tensor.dim() == 3 else 1

        # 处理值范围：检查是 0-1 还是 0-255
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
            # float 类型，检查值范围
            if tensor.max() <= 1.0:
                # 值在 0-1 范围，需要转换到 0-255
                tensor = (tensor * 255).clamp(0, 255)
            tensor = tensor.to(torch.uint8)
        elif tensor.dtype == torch.uint8:
            # 已经是 uint8，无需处理
            pass
        else:
            tensor = tensor.to(torch.uint8)

        # 转换 numpy 并确保正确的形状
        img_np = tensor.cpu().numpy()

        # 如果是 HWC 格式 (ComfyUI 标准格式)，直接使用
        # 如果是 CHW 格式，需要转换
        if img_np.shape[0] == channels and channels in [1, 3, 4]:
            img_np = img_np.transpose(1, 2, 0)

        # 处理单通道图像
        if channels == 1:
            img_np = img_np.squeeze(-1)

        return Image.fromarray(img_np)

    except Exception as e:
        raise ValueError(
            f"无法转换图像 tensor: {e}, shape: {tensor.shape}, dtype: {tensor.dtype}"
        )


def call_vlm_api(
    api_url,
    model_name,
    api_key,
    images,
    prompt,
    temperature=0.7,
    max_tokens=2048,
    system_prompt=None,
):
    """Call vLLM API for vision language model inference"""
    if not validate_url(api_url):
        raise ValueError(f"无效的API地址: {api_url}")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    image_contents = []
    for img in images:
        try:
            if isinstance(img, torch.Tensor):
                pil_img = tensor_to_pil(img)
            else:
                pil_img = img
            image_contents.append(f"data:image/png;base64,{pil2base64(pil_img)}")
        except Exception as e:
            raise ValueError(f"图像转换失败: {e}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    if image_contents:
        user_content.extend(
            [
                ({"type": "image_url", "image_url": {"url": img}})
                for img in image_contents
            ]
        )
    user_content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{api_url}/v1/chat/completions", headers=headers, json=payload, timeout=300
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"无法连接到API服务: {api_url}，请检查服务是否启动")
    except requests.exceptions.Timeout:
        raise TimeoutError("API请求超时，请检查网络连接或增加超时时间")
    except Exception as e:
        raise RuntimeError(f"API调用失败: {e}")


def generate_split_prompts(image_prompts, num_images):
    """Generate split-shot video prompts with camera movements"""
    movements = [
        "固定特写",
        "缓慢推近",
        "缓慢拉远",
        "缓慢左移",
        "缓慢右移",
        "上升鸟瞰",
        "下降俯视",
        "横移",
        "推拉摇移",
        "旋转",
    ]

    prompts = []
    for i, img_prompt in enumerate(image_prompts):
        movement = movements[i % len(movements)]
        prompts.append(f"镜头{i + 1}: {img_prompt}，运镜：{movement}")

    return "\n".join(prompts)


def generate_continuous_prompts(image_prompts):
    """Generate continuous video script with transitions"""
    transitions = ["然后", "接着", "随后", "随即", "紧接着", "画面切换到", "镜头转移到"]

    segments = []
    for i, img_prompt in enumerate(image_prompts):
        if i == 0:
            segment = f"画面开始于：{img_prompt}"
        elif i == len(image_prompts) - 1:
            segment = f"最终画面：{img_prompt}，然后淡出结束"
        else:
            transition = transitions[(i - 1) % len(transitions)]
            segment = f"{transition}：{img_prompt}"
        segments.append(segment)

    continuous = "，".join(segments[:2])
    for i in range(2, len(segments)):
        continuous += f"。"
        if segments[i].startswith("最终"):
            continuous += segments[i]
        else:
            continuous += segments[i].lower()

    if len(segments) > 1:
        continuous = segments[0] + "，"
        for i in range(1, len(segments)):
            if i < len(segments) - 1:
                continuous += (
                    f"{transitions[(i - 1) % len(transitions)]} {segments[i]}，"
                )
            else:
                continuous += f"最后 {segments[i]}"

    return continuous


class VLMImageToVideoPrompt:
    """VLM Image to Video Prompt Generator Node"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "http://localhost:8000"}),
                "model_name": ("STRING", {"default": "Qwen2-VL-72B-Instruct"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image_prompt": (
                    "STRING",
                    {
                        "default": "请仔细描述这张图片的所有细节，包括：主体、背景、光线、色彩、氛围、构图等。请用简洁的中文描述，50-100字。",
                        "multiline": True,
                    },
                ),
                "video_prompt": (
                    "STRING",
                    {
                        "default": "请根据以下多张图片的顺序，写出一个连续的视频镜头脚本。要求：1）保持画面之间的连续性和逻辑性；2）添加合理的镜头运动描述；3）描述画面之间的过渡效果。请用中文输出。",
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 8192, "step": 1},
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "video_prompts_split", "video_prompts_continuous")
    FUNCTION = "generate_prompts"
    CATEGORY = "VLMVideo"
    DESCRIPTION = "使用VLM模型反推图像提示词并生成视频提示词"

    def generate_prompts(
        self,
        api_url,
        model_name,
        api_key,
        system_prompt,
        image_prompt,
        video_prompt,
        temperature,
        max_tokens,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
    ):

        if not validate_url(api_url):
            return ("API地址格式无效，请输入以 http:// 或 https:// 开头的地址", "", "")

        images = []
        for img in [image1, image2, image3, image4, image5, image6]:
            if img is not None:
                images.append(img)

        if not images:
            return ("请至少输入一张图片", "", "")

        if not model_name:
            return ("请输入模型名称", "", "")

        sys_prompt = system_prompt if system_prompt else None
        img_prompt = (
            image_prompt
            if image_prompt
            else "请仔细描述这张图片的所有细节，包括：主体、背景、光线、色彩、氛围、构图等。请用简洁的中文描述，50-100字。"
        )

        image_prompts = []

        for i, img in enumerate(images):
            try:
                prompt = call_vlm_api(
                    api_url,
                    model_name,
                    api_key,
                    [img],
                    img_prompt,
                    temperature,
                    max_tokens,
                    sys_prompt,
                )
                image_prompts.append(f"图片{i + 1}: {prompt}")
            except Exception as e:
                image_prompts.append(f"图片{i + 1}: [识别失败 - {str(e)}]")

        image_prompts_str = "\n".join(image_prompts)

        vid_prompt = (
            video_prompt
            if video_prompt
            else "请根据以下多张图片的顺序，写出一个连续的视频镜头脚本。要求：1）保持画面之间的连续性和逻辑性；2）添加合理的镜头运动描述；3）描述画面之间的过渡效果。请用中文输出。"
        )

        try:
            video_result = call_vlm_api(
                api_url,
                model_name,
                api_key,
                images,
                f"{vid_prompt}\n\n图片顺序：\n{image_prompts_str}",
                temperature,
                max_tokens,
                sys_prompt,
            )
            video_prompts_continuous = video_result
        except Exception as e:
            successful_prompts = [
                p.split(": ", 1)[1] if ": " in p else p
                for p in image_prompts
                if "[识别失败" not in p
            ]
            if successful_prompts:
                video_prompts_continuous = generate_continuous_prompts(
                    successful_prompts
                )
            else:
                video_prompts_continuous = f"[生成失败: {str(e)}]"

        video_prompts_split = generate_split_prompts(
            [
                p.split(": ", 1)[1] if ": " in p else p
                for p in image_prompts
                if "[识别失败" not in p
            ],
            len(images),
        )

        return (image_prompts_str, video_prompts_split, video_prompts_continuous)


NODE_CLASS_MAPPINGS["VLMImageToVideoPrompt"] = VLMImageToVideoPrompt
NODE_DISPLAY_NAME_MAPPINGS["VLMImageToVideoPrompt"] = "VLM图像转视频提示词"


class VLMSingleImagePrompt:
    """VLM Single Image to Prompt Node"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "http://localhost:8000"}),
                "model_name": ("STRING", {"default": "Qwen2-VL-72B-Instruct"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": (
                    "STRING",
                    {"default": "请详细描述这张图片的所有细节", "multiline": True},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 8192, "step": 1},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "VLMVideo"
    DESCRIPTION = "使用VLM模型单张图像反推提示词"

    def generate_prompt(
        self,
        api_url,
        model_name,
        api_key,
        system_prompt,
        prompt,
        temperature,
        max_tokens,
        image=None,
    ):
        if not validate_url(api_url):
            return ("API地址格式无效，请输入以 http:// 或 https:// 开头的地址",)

        if not model_name:
            return ("请输入模型名称",)

        if image is None:
            return ("请输入图片",)

        sys_prompt = system_prompt if system_prompt else None

        try:
            result = call_vlm_api(
                api_url,
                model_name,
                api_key,
                [image],
                prompt,
                temperature,
                max_tokens,
                sys_prompt,
            )
            return (result,)
        except Exception as e:
            return (f"错误: {str(e)}",)


NODE_CLASS_MAPPINGS["VLMSingleImagePrompt"] = VLMSingleImagePrompt
NODE_DISPLAY_NAME_MAPPINGS["VLMSingleImagePrompt"] = "VLM单图提示词反推"


class VideoPromptEnhancer:
    """Video Prompt Enhancer Node"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "http://localhost:8000"}),
                "model_name": ("STRING", {"default": "Qwen2-VL-72B-Instruct"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompts": ("STRING", {"multiline": True}),
                "enhance_type": (
                    ["分镜头模式", "连续镜头模式", "运镜增强", "风格统一"],
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 8192, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "enhance_prompts"
    CATEGORY = "VLMVideo"
    DESCRIPTION = "增强和优化视频提示词"

    def enhance_prompts(
        self,
        api_url,
        model_name,
        api_key,
        system_prompt,
        prompts,
        enhance_type,
        temperature,
        max_tokens,
    ):
        if not validate_url(api_url):
            return ("API地址格式无效，请输入以 http:// 或 https:// 开头的地址",)

        if not model_name:
            return ("请输入模型名称",)

        enhance_prompts_map = {
            "分镜头模式": "请将以下提示词整理成标准的分镜头脚本格式，每个镜头包含：镜头编号、画面描述、运镜方式。",
            "连续镜头模式": "请将以下提示词整理成一个连贯的视频脚本，添加过渡描述，使画面之间逻辑通顺。",
            "运镜增强": "请为以下每个画面添加专业的摄影运镜术语，如：推拉摇移、跟拍、甩镜头等。",
            "风格统一": "请统一以下提示词的视觉风格和语言风格，使其具有一致性。",
        }

        prompt = f"{enhance_prompts_map[enhance_type]}\n\n{prompts}"
        sys_prompt = system_prompt if system_prompt else None

        try:
            result = call_vlm_api(
                api_url,
                model_name,
                api_key,
                [],
                prompt,
                temperature,
                max_tokens,
                sys_prompt,
            )
            return (result,)
        except Exception as e:
            return (f"错误: {str(e)}",)


NODE_CLASS_MAPPINGS["VideoPromptEnhancer"] = VideoPromptEnhancer
NODE_DISPLAY_NAME_MAPPINGS["VideoPromptEnhancer"] = "视频提示词增强"


print("VLMVideoPrompts nodes loaded successfully!")
