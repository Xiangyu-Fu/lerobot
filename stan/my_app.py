# Hydra app
import hydra
from omegaconf import DictConfig, OmegaConf

# Torch
import torch
from torch import nn, Tensor
# from PIL import Image
import torchvision
from torchviz import make_dot
# from torchvision.transforms import ToPILImage

# Python
import math
import numpy as np
import einops
from pathlib import Path
from collections import deque

# Huggingface Lerobot, should replace with my own imports later
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.normalize import Normalize, Unnormalize

# my own imports
from stan.utils.utils import _replace_submodules, _make_noise_scheduler
from stan.conf.my_configuration_diffusion import MyDiffusionConfig

# class MyDiffusionConfig:
#     def __init__(self) -> None:
#         self.foo = "bar"


class MyDiffusionPolicy(
    nn.Module,
):
    def __init__(
            self, 
            config: MyDiffusionConfig | None = None, 
            dataset_stats: dict[str, dict[str, Tensor]] | None = None,  # 数据集统计信息
            ):
        super().__init__()
        if config is None:
            config = MyDiffusionConfig()  # Here we use the default config
        self.config = config

        # TODO: replace with my own implementation
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self._queues = None

        self.diffusion = MyDiffusionModel(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        # if self.use_env_state:
        #     self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Do something with the input batch
        # input shape: (36864, 96) = (64*2*3*96, 96)
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # Copy the batch to avoid modifying the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch) 
        loss = self.diffusion.compute_loss(batch)  # TODO
        # Return the output dictionary
        return {"loss": loss}


class MyDiffusionModel(nn.Module):
    def __init__(self, config: MyDiffusionConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        # config.input_shapes = {'observation.image': [3, 96, 96], 'observation.state': [2]}
        global_cond_dim = config.input_shapes["observation.state"][0] 
        num_images = len([k for k in config.input_shapes if k.startswith("observation.image")])
        self._use_images = False
        self._use_env_state = False
        if num_images > 0:
            self._use_images = True
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += config.diffusion_step_embed_dim  # 2+128

        # U-Net
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)  # TODO

        # Build the noise scheduler 
        # TODO: check the code
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is not None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        pass
        
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        n_obs_steps = batch["observation.state"].shape[1] # ？
        horizon = batch["action"].shape[1] 
        assert n_obs_steps == self.config.n_obs_steps
        assert horizon == self.config.horizon

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        output = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        
        dot = make_dot(output)
        dot.render("output", format="png")

        return torch.tensor(0.0)

    
# ================== Encoder ==================
class DiffusionRgbEncoder(nn.Module):
    '''Encode the image observation into a 1D feature vector.
    '''
    def __init__(self, config:MyDiffusionConfig):
        super().__init__()
        self.config = config
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.random_crop = self.center_crop
        else:
            self.do_crop = False

        # set up the vision backbone
        # getattr 获取动态属性， 等效于
        # backbone_model = torchvision.models.resnet18(pretrained=True)
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            pretrained=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        # 替换批归一化层为组归一化层
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # set up pooling and final layers
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
            )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SptialSoftmax(feature_map_shape, config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2  # 2 for x and y
        self.out = nn.Linear(self.feature_dim, config.diffusion_step_embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, C, H, W) image tensor with pixel values in [0, 1]
        Returns:
            (batch_size, diffusion_step_embed_dim) tensor
        """
        if self.do_crop:
            if self.training:
                x = self.random_crop(x)
            else:
                x = self.center_crop(x)  # Center crop during evaluation
            
        # Extract features
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x
    

class SptialSoftmax(nn.Module):
    def __init__(self, input_shape, num_keypoints=None):
        super().__init__()
        
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_keypoints is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_keypoints, kernel_size=1)
            self._out_c = num_keypoints
        else:
            self.nets = None
            self._out_c = self._in_c

        # 1. Create a meshgrid of the input shape
        pos_x, pos_y = np.meshgrid(np.linspace(-1, 1, self._in_w), np.linspace(-1, 1, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forware(self, features:Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        
        # [B, K, H, W] -> [B * K, H * W]
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2. Apply softmax to get the weights
        attention = torch.nn.functional.softmax(features, dim=1)
        # 3. Compute the weighted sum of the positions
        excepted_xy = attention @ self.pos_grid

        feature_keypoints = excepted_xy.reshape(-1, self._out_c, 2)

        return feature_keypoints

# ================== Unet ==================
class DiffusionConditionalUnet1d(nn.Module):
    '''实现了一个一维卷积的条件 U-Net 模型，并结合了 FiLM(Feature-wise Linear Modulation)调节,
    用于扩散模型的应用场景。U-Net 是一种常见的网络结构，通常用于图像分割任务，但这里使用的是一维卷积，
    因此适用于时间序列或类似的任务。
    该模型采用编码器-解码器结构，利用 skip connections来保留输入的高分辨率特征,并结合扩散过程的条件信息进行建模。
    '''
    def __init__(self, config:MyDiffusionConfig, global_cond_dim:int):
        super().__init__()
        self.config = config

        # 1. 扩散步骤编码器, 将时间步长转化为一个与输入特征兼容的嵌入
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # 2. FiLM条件维度 388 = 128 + 260
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim  # 这里的conditional dimension是什么？

        # 3. 下采样编码器, len(config.down_dims) = 3
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # 4. unet 编码器模块
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 5. 中间处理模块
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # 6. 上采样解码器
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        
        # 7. 最终卷积层
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        )
    
    def forward(self, x:Tensor, timestep:Tensor|int, global_cond=None) -> Tensor:
        """
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        # Encode the diffusion step.
        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionSinusoidalPosEmb(nn.Module):
    """这个类实现了 1D 正弦位置嵌入，用于为时间序列数据提供位置信息。
    通过正弦和余弦函数生成嵌入向量，并结合不同频率的缩放因子来区分不同的位置。
    嵌入维度 dim 被分为两部分：一部分用于正弦嵌入，另一部分用于余弦嵌入。
    TODO: look in detail about how the embedding is generated.
    Args:
        dim (int): 嵌入向量的维度
    Output:
        emb (Tensor): 位置嵌入向量, shape 为 (B, T, dim)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish
    这个模块适用于时间序列数据或一维特征的处理场景，包含了一个卷积层、组归一化层和 Mish 激活函数。"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """DiffusionConditionalResidualBlock1d 实现了一个带有条件调节(FiLM)的 1D 卷积残差块。
    该结构适用于处理时间序列或一维数据，允许通过条件张量对特征进行灵活的调节。
    - FiLM 调节：可以调节每个通道的缩放(scale)和偏移(bias)，使得模型能够根据条件信息动态调整特征的分布。
    - 残差连接：保证了梯度的稳定传递，并提供了更好的训练表现。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels # if Film, then 2*out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

# ================== Main ==================
@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(config : DictConfig) -> None:
    # print(OmegaConf.to_yaml(config))
    output_directory = Path("outputs/test_outputs/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 5000
    device = torch.device("cuda")
    log_freq = 250

    '''
    Dataset:
        - input shape: (batch_size, 2, C, H, W) # 每个样本的图像观测会包含两个不同时间点的信息
        - state shape: (batch_size, 2, 2)
        - action shape: (batch_size, 16, 2)
    '''

    # Set up the dataset.
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4, # Set to 0 if you're running this on Windows. default is 4.
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Show some examples of how to use the dataset
    print(f"Dataset size: {len(dataset)}")
    example_batch = next(iter(dataloader))
    print("Example batch keys:", example_batch.keys())

    # Let's visualize a few samples from the batch
    # Assuming 'observation.image' and 'action' are keys in the dataset
    images = example_batch["observation.image"]
    states = example_batch["observation.state"]
    actions = example_batch["action"]

    print(f"Images shape: {images.shape}")  # Should be (batch_size, 2, C, H, W) -> 2 timestamps per sample
    print(f"States shape: {states.shape}")  # Should be (batch_size, 2, state_dim)
    print(f"Actions shape: {actions.shape}")  # Should be (batch_size, action_dim)

    # # Visualize the first sample's images from the batch
    # # We use torchvision and PIL to visualize images without using matplotlib
    # to_pil = ToPILImage()

    # for i in range(2):  # There are 2 timestamps for the images
    #     image = images[0, i].cpu()  # Get the first sample, first/second image
    #     image_pil = to_pil(image)  # Convert tensor to PIL image
    #     image_pil.show(title=f"Timestamp {i}: Image")  # Display the image using PIL's native viewer

    # # Print state and action values for the first sample
    # print("States for the first sample:", states[0].cpu().numpy())
    # print("Actions for the first sample:", actions[0].cpu().numpy())

    # Set up the the diffusion model.
    config = MyDiffusionConfig()
    policy = MyDiffusionPolicy(config, dataset_stats=dataset.stats)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # batch.items() 返回键值对
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break



if __name__ == "__main__":
    my_app()