import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import USE_PEFT_BACKEND, is_torch_xla_available, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    """
    计算平移因子，用于调整时间步的分布。

    参数:
        image_seq_len (int): 图像序列的长度。
        base_seq_len (int): 基础序列长度，默认为256。
        max_seq_len (int): 最大序列长度，默认为4096。
        base_shift (float): 基础平移量，默认为0.5。
        max_shift (float): 最大平移量，默认为1.16。

    返回:
        float: 计算得到的平移因子。
    """
    # 计算斜率
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    # 计算截距
    b = base_shift - m * base_seq_len
    # 计算平移因子
    mu = image_seq_len * m + b
    # 返回平移因子
    return mu


def prepare_latent_image_ids_2(height, width, device, dtype):
    """
    准备潜在图像的ID张量，用于位置编码。

    参数:
        height (int): 图像的高度。
        width (int): 图像的宽度。
        device (torch.device): 张量所在的设备。
        dtype (torch.dtype): 张量的数据类型。

    返回:
        torch.Tensor: 准备好的潜在图像ID张量，形状为 (height//2, width//2, 3)。
    """
    # 初始化潜在图像ID张量，形状为 (height//2, width//2, 3)
    latent_image_ids = torch.zeros(height//2, width//2, 3, device=device, dtype=dtype)
    # 为第二维添加y坐标
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height//2, device=device)[:, None]
    # 为第三维添加x坐标
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width//2, device=device)[None, :]
    # 返回潜在图像ID张量
    return latent_image_ids


def position_encoding_clone(batch_size, original_height, original_width, device, dtype):
    """
    克隆位置编码张量，用于潜在图像。

    参数:
        batch_size (int): 批量大小。
        original_height (int): 原始图像的高度。
        original_width (int): 原始图像的宽度。
        device (torch.device): 张量所在的设备。
        dtype (torch.dtype): 张量的数据类型。

    返回:
        torch.Tensor: 克隆后的位置编码张量，形状为 (batch_size, latent_image_id_height * latent_image_id_width, 6)。
    """
    # 准备潜在图像ID张量
    latent_image_ids = prepare_latent_image_ids_2(original_height, original_width, device, dtype)
    # 获取潜在图像ID张量的形状
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    # 重塑张量形状为 (latent_image_id_height * latent_image_id_width, latent_image_id_channels)
    latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
    # 复制潜在图像ID张量
    cond_latent_image_ids = latent_image_ids
    # 在最后一个维度上拼接，形状变为 (latent_image_id_height * latent_image_id_width, 6)
    latent_image_ids = torch.concat([latent_image_ids, cond_latent_image_ids], dim=-2)
    # 返回克隆后的位置编码张量
    return latent_image_ids


# 从diffusers库的stable_diffusion_img2img管道中复制的方法，用于检索潜在向量
def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    """
    从编码器输出中检索潜在向量。

    参数:
        encoder_output (torch.Tensor): 编码器输出张量。
        generator (Optional[torch.Generator]): 可选的随机数生成器，用于控制采样。
        sample_mode (str): 采样模式，可以是 "sample" 或 "argmax"，默认为 "sample"。

    返回:
        torch.Tensor: 检索到的潜在向量。

    异常:
        AttributeError: 如果提供的编码器输出中不存在潜在向量或潜在分布。
    """
    # 从潜在分布中采样
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    # 获取潜在分布的模式
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        # 返回潜在向量
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# 从diffusers库的stable_diffusion管道中复制的方法，用于检索时间步
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    调用调度器的 `set_timesteps` 方法并从调度器中检索时间步。处理自定义时间步。任何关键字参数都将传递给 `scheduler.set_timesteps`。

    参数:
        scheduler (`SchedulerMixin`): 要从中获取时间步的调度器。
        num_inference_steps (`int`, *可选*): 使用预训练模型生成样本时使用的扩散步数。如果使用，则 `timesteps` 必须为 `None`。
        device (`str` 或 `torch.device`, *可选*): 时间步应移动到的设备。如果 `None`，则时间步不移动。
        timesteps (`List[int]`, *可选*): 用于覆盖调度器的时间步间距策略的自定义时间步。如果传递了 `timesteps`，则 `num_inference_steps` 和 `sigmas` 必须为 `None`。
        sigmas (`List[float]`, *可选*): 用于覆盖调度器的时间步间距策略的自定义sigma。如果传递了 `sigmas`，则 `num_inference_steps` 和 `timesteps` 必须为 `None`。

    返回:
        `Tuple[torch.Tensor, int]`: 一个元组，其中第一个元素是调度器的时间步调度，第二个元素是推理步数。
    """
    # 检查是否同时传递了timesteps和sigmas
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    # 检查调度器是否支持自定义时间步
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # 设置时间步
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        # 获取时间步
        timesteps = scheduler.timesteps
        # 获取推理步数
        num_inference_steps = len(timesteps)

    # 检查调度器是否支持自定义sigma
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # 设置sigma
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        # 获取时间步
        timesteps = scheduler.timesteps
        # 获取推理步数
        num_inference_steps = len(timesteps)

    else:
        # 设置推理步数
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        # 获取时间步
        timesteps = scheduler.timesteps

    # 返回时间步和推理步数
    return timesteps, num_inference_steps


class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin):
    """
    FluxPipeline 类，用于文本生成图像的管道。

    参数:
        transformer (`FluxTransformer2DModel`):
            条件Transformer（MMDiT）架构，用于对编码后的图像潜在向量进行去噪。
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            与 `transformer` 结合使用的调度器，用于对编码后的图像潜在向量进行去噪。
        vae (`AutoencoderKL`):
            可变自动编码器（VAE）模型，用于将图像编码和解码为潜在表示。
        text_encoder (`CLIPTextModel`):
            CLIP文本编码器，用于将文本转换为嵌入向量。
        text_encoder_2 (`T5EncoderModel`):
            T5文本编码器，用于将文本转换为嵌入向量。
        tokenizer (`CLIPTokenizer`):
            CLIP分词器，用于将文本转换为标记。
        tokenizer_2 (`T5TokenizerFast`):
            T5快速分词器，用于将文本转换为标记。
    """
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    # 可选组件列表
    _optional_components = []
    # 回调张量输入列表
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
            self,
            scheduler: FlowMatchEulerDiscreteScheduler,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            text_encoder_2: T5EncoderModel,
            tokenizer_2: T5TokenizerFast,
            transformer: FluxTransformer2DModel,
    ):
        super().__init__()
        # 注册模块
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        # 计算VAE的缩放因子，如果存在VAE，则为2的块输出通道数长度次方；否则为16
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        # 初始化图像处理器，传入VAE的缩放因子
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # 获取分词器的最大长度，如果存在分词器，则为分词器的最大长度；否则为77
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        # 设置默认的样本大小为64
        self.default_sample_size = 64

    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_images_per_prompt: int = 1,
            max_sequence_length: int = 512,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        """
        使用 T5 分词器和文本编码器获取提示嵌入。

        参数:
            prompt (str 或 List[str]): 输入提示。
            num_images_per_prompt (int): 每个提示生成的图像数量，默认为1。
            max_sequence_length (int): 最大序列长度，默认为512。
            device (torch.device, optional): 设备，默认为None。
            dtype (torch.dtype, optional): 数据类型，默认为None。

        返回:
            torch.Tensor: 提示嵌入张量。
        """
        # 获取设备，如果未指定，则使用执行设备
        device = device or self._execution_device
        # 获取数据类型，如果未指定，则使用文本编码器的数据类型
        dtype = dtype or self.text_encoder.dtype

        # 如果提示是字符串，则转换为列表
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 获取批量大小
        batch_size = len(prompt)

        # 使用 T5 分词器对提示进行编码
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        # 获取输入ID
        text_input_ids = text_inputs.input_ids
        # 获取未截断的输入ID
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        # 检查是否有截断，并记录警告
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # 使用 T5 文本编码器生成嵌入
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        # 获取文本编码器的数据类型
        dtype = self.text_encoder_2.dtype
        # 将嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # 获取嵌入的形状
        _, seq_len, _ = prompt_embeds.shape

        # 为每个生成的图像重复文本嵌入和注意力掩码，使用 MPS 友好的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 返回提示嵌入
        return prompt_embeds

    def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            device: Optional[torch.device] = None,
    ):
        """
        使用 CLIP 分词器和文本编码器获取提示嵌入。

        参数:
            prompt (str 或 List[str]): 输入提示。
            num_images_per_prompt (int): 每个提示生成的图像数量，默认为1。
            device (torch.device, optional): 设备，默认为None。

        返回:
            torch.Tensor: 提示嵌入张量。
        """
        # 获取设备，如果未指定，则使用执行设备
        device = device or self._execution_device

        # 如果提示是字符串，则转换为列表
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # 获取批量大小
        batch_size = len(prompt)

        # 使用 CLIP 分词器对提示进行编码
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        # 获取输入ID
        text_input_ids = text_inputs.input_ids
        # 获取未截断的输入ID
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        # 检查是否有截断，并记录警告
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        # 使用 CLIP 文本编码器生成嵌入
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # 使用 CLIPTextModel 的池化输出
        prompt_embeds = prompt_embeds.pooler_output
        # 将嵌入转换为指定的数据类型和设备
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # 为每个生成的图像重复文本嵌入，使用 MPS 友好的方法
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # 返回提示嵌入
        return prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            prompt_2: Union[str, List[str]],
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            max_sequence_length: int = 512,
            lora_scale: Optional[float] = None,
    ):
        """
        对提示进行编码，生成文本嵌入。

        参数:
            prompt (str 或 List[str], *可选*): 要编码的提示。
            prompt_2 (str 或 List[str], *可选*): 要发送给 `tokenizer_2` 和 `text_encoder_2` 的提示。如果未定义，则所有文本编码器都使用 `prompt`。
            device (torch.device): torch 设备。
            num_images_per_prompt (int): 每个提示生成的图像数量。
            prompt_embeds (torch.FloatTensor, *可选*): 预生成的文本嵌入。用于轻松调整文本输入，例如提示权重。如果未提供，将从 `prompt` 输入参数生成文本嵌入。
            pooled_prompt_embeds (torch.FloatTensor, *可选*): 预生成的池化文本嵌入。用于轻松调整文本输入，例如提示权重。如果未提供，将从 `prompt` 输入参数生成池化文本嵌入。
            max_sequence_length (int): 最大序列长度，默认为 512。
            lora_scale (float, *可选*): 如果加载了 LoRA 层，将应用于文本编码器所有 LoRA 层的 LoRA 缩放因子。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 返回文本嵌入、池化文本嵌入和文本 ID 张量。
        """
        device = device or self._execution_device

        # 设置 LoRA 缩放因子，以便文本编码器的补丁 LoRA 函数可以正确访问它
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # 动态调整 LoRA 缩放因子
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        # 如果提示是字符串，则转换为列表
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            # 如果未提供 prompt_2，则使用 prompt
            prompt_2 = prompt_2 or prompt
            # 如果 prompt_2 是字符串，则转换为列表
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # 仅使用 CLIPTextModel 的池化提示输出
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # 通过缩放回 LoRA 层来检索原始缩放因子
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # 通过缩放回 LoRA 层来检索原始缩放因子
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        # 获取文本编码器的数据类型
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        # 创建文本 ID 张量，形状为 (seq_len, 3)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        # 返回文本嵌入、池化文本嵌入和文本 ID 张量
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # 从 diffusers 库的 stable_diffusion_3_inpaint 管道中复制的方法，用于编码 VAE 图像
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        """
        对图像进行编码，生成潜在向量。

        参数:
            image (torch.Tensor): 输入图像张量。
            generator (torch.Generator): 随机数生成器，用于控制采样。

        返回:
            torch.Tensor: 编码后的潜在向量。
        """
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            # 拼接潜在向量
            image_latents = torch.cat(image_latents, dim=0)
        else:
            # 编码图像并检索潜在向量
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        # 调整潜在向量
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 返回调整后的潜在向量
        return image_latents

    def check_inputs(
            self,
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=None,
    ):
        """
        检查输入参数的有效性。

        参数:
            prompt (str 或 List[str], *可选*): 输入提示。
            prompt_2 (str 或 List[str], *可选*): 输入提示2。
            height (int): 图像高度。
            width (int): 图像宽度。
            prompt_embeds (torch.FloatTensor, *可选*): 预生成的文本嵌入。
            pooled_prompt_embeds (torch.FloatTensor, *可选*): 预生成的池化文本嵌入。
            callback_on_step_end_tensor_inputs (List[str], *可选*): 回调函数在步骤结束时输入的张量名称列表。
            max_sequence_length (int, *可选*): 最大序列长度。
        """
        if height % 8 != 0 or width % 8 != 0:
            # 检查高度和宽度是否被8整除
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            # 检查回调函数输入的张量名称是否有效
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            # 检查是否同时提供了提示和提示嵌入
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            # 检查是否同时提供了提示2和提示嵌入
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            # 检查是否提供了提示或提示嵌入
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            # 检查提示类型
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            # 检查提示2类型
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            # 检查是否同时提供了提示嵌入和池化提示嵌入
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            # 检查最大序列长度是否超过512
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        """
        准备潜在图像的ID张量，用于位置编码。

        参数:
            batch_size (int): 批量大小。
            height (int): 图像的高度。
            width (int): 图像的宽度。
            device (torch.device): 张量所在的设备。
            dtype (torch.dtype): 张量的数据类型。

        返回:
            torch.Tensor: 准备好的潜在图像ID张量，形状为 (height//2 * width//2, 3)。
        """
        # 初始化潜在图像ID张量，形状为 (height//2, width//2, 3)
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        # 为第二维添加y坐标
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        # 为第三维添加x坐标
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
        # 获取潜在图像ID张量的形状
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        # 重塑张量形状为 (latent_image_id_height * latent_image_id_width, latent_image_id_channels)
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        # 返回转移到指定设备和数据类型的潜在图像ID张量
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """
        打包潜在向量，以便进行后续处理。

        参数:
            latents (torch.Tensor): 输入的潜在向量张量。
            batch_size (int): 批量大小。
            num_channels_latents (int): 潜在向量的通道数。
            height (int): 图像的高度。
            width (int): 图像的宽度。

        返回:
            torch.Tensor: 打包后的潜在向量张量，形状为 (batch_size, (height//2) * (width//2), num_channels_latents * 4)。
        """
        # 重塑张量形状
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        # 调整维度顺序
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        # 重塑为 (batch_size, (height//2) * (width//2), num_channels_latents * 4)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        # 返回打包后的潜在向量
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """
        解包潜在向量，恢复原始形状。

        参数:
            latents (torch.Tensor): 输入的潜在向量张量。
            height (int): 图像的高度。
            width (int): 图像的宽度。
            vae_scale_factor (int): VAE的缩放因子。

        返回:
            torch.Tensor: 解包后的潜在向量张量，形状为 (batch_size, channels//4, height * 2, width * 2)。
        """
        # 获取张量的形状
        batch_size, num_patches, channels = latents.shape

        # 计算调整后的高度
        height = height // vae_scale_factor
        # 计算调整后的宽度
        width = width // vae_scale_factor

        # 重塑张量形状
        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        # 调整维度顺序
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        # 重塑为 (batch_size, channels//4, height * 2, width * 2)
        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        # 返回解包后的潜在向量
        return latents

    def enable_vae_slicing(self):
        """
        启用切片VAE解码。当启用此选项时，VAE会将输入张量分割成多个切片，分步计算解码。这有助于节省内存并允许更大的批量大小。
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """
        禁用切片VAE解码。如果之前启用了 `enable_vae_slicing`，此方法将恢复为单步计算解码。
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        """
        启用平铺VAE解码。当启用此选项时，VAE会将输入张量分割成多个平铺块，分步计算解码和编码。这有助于节省大量内存并允许处理更大的图像。
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        """
        禁用平铺VAE解码。如果之前启用了 `enable_vae_tiling`，此方法将恢复为单步计算解码。
        """
        self.vae.disable_tiling()

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
            condition_image=None,
    ):
        """
        准备潜在向量，用于模型的前向传播。

        参数:
            batch_size (int): 批量大小。
            num_channels_latents (int): 潜在向量的通道数。
            height (int): 图像的高度。
            width (int): 图像的宽度。
            dtype (torch.dtype): 数据类型。
            device (torch.device): 设备。
            generator (Optional[torch.Generator]): 随机数生成器。
            latents (Optional[torch.Tensor]): 预定义的潜在向量，默认为None。
            condition_image (Optional[torch.Tensor]): 条件图像，默认为None。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 返回打包后的潜在向量、潜在图像ID张量、掩码和条件潜在向量。
        """
        # 计算调整后的高度
        height = 2 * (int(height) // self.vae_scale_factor)  
        # 计算调整后的宽度
        width = 2 * (int(width) // self.vae_scale_factor)

        # 定义张量形状，例如 (1, 16, 106, 80)
        shape = (batch_size, num_channels_latents, height, width)  # 1 16 106 80

        if latents is not None:
            # 准备潜在图像ID张量
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            # 返回潜在向量和潜在图像ID张量
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            # 检查生成器列表长度是否与批量大小匹配
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if condition_image is not None:
            # 将条件图像转移到指定设备和数据类型
            condition_image = condition_image.to(device=device, dtype=dtype)
            # 对条件图像进行编码，生成潜在向量
            image_latents = self._encode_vae_image(image=condition_image, generator=generator)
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # 如果批量大小大于条件图像的批量大小，并且可以整除，则扩展条件图像的潜在向量
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                # 如果无法整除，则抛出异常
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                # 否则，拼接条件图像的潜在向量
                image_latents = torch.cat([image_latents], dim=0)

        # 生成随机潜在向量
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  
        # 打包潜在向量
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        # 打包条件图像的潜在向量
        cond_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        # 拼接潜在向量和条件潜在向量
        latents = torch.concat([latents, cond_latents], dim=-2)

        # 添加位置编码
        latent_image_ids = position_encoding_clone(batch_size, height, width, device, dtype)  # add position

        # 创建全1掩码
        mask1 = torch.ones(shape, device=device, dtype=dtype)
        # 创建全0掩码
        mask2 = torch.zeros(shape, device=device, dtype=dtype)

        # 打包全1掩码
        mask1 = self._pack_latents(mask1, batch_size, num_channels_latents, height, width)  # 1 4096 64
        # 打包全0掩码
        mask2 = self._pack_latents(mask2, batch_size, num_channels_latents, height, width)  # 1 4096 64
        # 拼接掩码
        mask = torch.concat([mask1, mask2], dim=-2)

        # 返回打包后的潜在向量、潜在图像ID张量、掩码和条件潜在向量
        return latents, latent_image_ids, mask, cond_latents

    @property
    def guidance_scale(self):
        """
        获取指导缩放因子。

        Returns:
            float: 指导缩放因子。
        """
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        """
        获取联合注意力参数。

        Returns:
            dict: 联合注意力参数。
        """
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        """
        获取时间步的数量。

        Returns:
            int: 时间步的数量。
        """
        return self._num_timesteps

    @property
    def interrupt(self):
        """
        获取中断标志。

        Returns:
            bool: 中断标志。
        """
        return self._interrupt

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            timesteps: List[int] = None,
            guidance_scale: float = 3.5,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 512,
            condition_image=None,
    ):
        """
        执行文本生成图像的推理过程。

        参数:
            prompt (str 或 List[str]): 输入提示。
            prompt_2 (str 或 List[str], optional): 第二个输入提示。
            height (int, optional): 图像的高度。
            width (int, optional): 图像的宽度。
            num_inference_steps (int): 推理步数，默认为28。
            timesteps (List[int], optional): 自定义时间步列表。
            guidance_scale (float): 指导缩放因子，默认为3.5。
            num_images_per_prompt (int, optional): 每个提示生成的图像数量，默认为1。
            generator (torch.Generator 或 List[torch.Generator], optional): 随机数生成器。
            latents (torch.FloatTensor, optional): 预定义的潜在向量。
            prompt_embeds (torch.FloatTensor, optional): 预生成的文本嵌入。
            pooled_prompt_embeds (torch.FloatTensor, optional): 预生成的池化文本嵌入。
            output_type (str, optional): 输出类型，默认为 "pil"。
            return_dict (bool): 是否返回字典，默认为True。
            joint_attention_kwargs (dict, optional): 联合注意力参数。
            callback_on_step_end (Callable, optional): 回调函数，在每一步结束时调用。
            callback_on_step_end_tensor_inputs (List[str]): 回调函数在每一步结束时输入的张量名称列表，默认为 ["latents"]。
            max_sequence_length (int): 最大序列长度，默认为512。
            condition_image (torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image], optional): 条件图像。

        返回:
            FluxPipelineOutput 或 Tuple[PIL.Image.Image]: 返回生成的图像或包含图像的字典。
        """
        # 如果未指定高度，则使用默认样本大小乘以VAE缩放因子
        height = height or self.default_sample_size * self.vae_scale_factor
        # 如果未指定宽度，则使用默认样本大小乘以VAE缩放因子
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. 检查输入。如果不正确，则抛出错误
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        
        # 设置指导缩放因子
        self._guidance_scale = guidance_scale
        # 设置联合注意力参数
        self._joint_attention_kwargs = joint_attention_kwargs
        # 重置中断标志
        self._interrupt = False
        
        # 预处理条件图像
        condition_image = self.image_processor.preprocess(condition_image, height=height, width=width)
        # 将条件图像转换为浮点类型
        condition_image = condition_image.to(dtype=torch.float32)

        # 2. 定义调用参数
        if prompt is not None and isinstance(prompt, str):
            # 如果提示是字符串，则批量大小为1
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            # 如果提示是列表，则批量大小为列表长度
            batch_size = len(prompt)
        else:
            # 否则，批量大小为提示嵌入的第一个维度
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 获取LoRA缩放因子
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(   # 对提示进行编码，生成文本嵌入
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. 准备潜在变量
        num_channels_latents = self.transformer.config.in_channels // 4  # 计算潜在向量的通道数，假设为16
        # 准备潜在向量
        latents, latent_image_ids, mask, cond_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            condition_image
        )
        # 克隆潜在向量以保存干净的版本
        clean_latents = latents.clone()  

        # 5. 准备时间步
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) # 生成sigma值
        # 获取图像序列长度
        image_seq_len = latents.shape[1]
        # 计算平移因子
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        # 获取时间步
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        # 计算预热步数
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # 设置时间步数量
        self._num_timesteps = len(timesteps)

        # 处理指导
        if self.transformer.config.guidance_embeds:
            # 创建指导张量
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            # 扩展到批量大小
            guidance = guidance.expand(latents.shape[0])
        else:
            # 如果不需要指导，则为None
            guidance = None

        # 6. 去噪循环
        with self.progress_bar(total=num_inference_steps) as progress_bar:  # 使用进度条
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    # 如果中断，则跳过
                    continue

                # 以与ONNX/Core ML兼容的方式广播到批量维度
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(  # 前向传播，获取噪声预测
                    hidden_states=latents,  # 潜在向量
                    timestep=timestep / 1000,  # 时间步
                    guidance=guidance,  # 指导
                    pooled_projections=pooled_prompt_embeds,  # 池化提示嵌入
                    encoder_hidden_states=prompt_embeds,  # 提示嵌入
                    txt_ids=text_ids,  # 文本ID
                    img_ids=latent_image_ids,  # 潜在图像ID
                    joint_attention_kwargs=self.joint_attention_kwargs,  # 联合注意力参数
                    return_dict=False,  # 不返回字典
                )[0]

                # 计算前一个噪声样本 x_t -> x_t-1
                latents_dtype = latents.dtype
                # 应用调度器步骤
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # 应用掩码
                latents = latents * mask + clean_latents * (1 - mask)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    # 准备回调参数
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    # 调用回调函数
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # 获取回调输出的潜在向量
                    latents = callback_outputs.pop("latents", latents)
                    # 获取回调输出的提示嵌入
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # 调用回调（如果提供）
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    # 更新进度条
                    progress_bar.update()

                if XLA_AVAILABLE:
                    # 如果使用XLA，则标记步骤
                    xm.mark_step()

        if output_type == "latent":
            # 如果输出类型为潜在向量，则直接返回潜在向量
            image = latents

        else:
            # 解包潜在向量
            latents = self._unpack_latents(latents[:,:latents.shape[-2]-cond_latents.shape[-2],:], height, width, self.vae_scale_factor)
            # 调整潜在向量
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            # 解码潜在向量生成图像
            image = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
            # 后处理图像
            image = self.image_processor.postprocess(image, output_type=output_type)

        # 卸载所有模型
        self.maybe_free_model_hooks()

        if not return_dict:
            # 如果不返回字典，则返回图像
            return (image,)

        # 返回包含图像的FluxPipelineOutput对象
        return FluxPipelineOutput(images=image)
