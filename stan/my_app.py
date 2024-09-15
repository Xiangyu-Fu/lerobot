# Hydra app
import hydra
from omegaconf import DictConfig, OmegaConf

# Torch
import torch
from torch import nn, Tensor
# from PIL import Image
import torchvision
# from torchvision.transforms import ToPILImage

# Python
import numpy as np
from pathlib import Path
from collections import deque

# Huggingface Lerobot, should replace with my own imports later
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# my own imports
from stan.utils.utils import _replace_submodules
from stan.conf.my_configuration_diffusion import MyDiffusionConfig

# class MyDiffusionConfig:
#     def __init__(self) -> None:
#         self.foo = "bar"


class MyDiffusionPolicy(
    nn.Module,
):
    def __init__(
            self, 
            config: MyDiffusionConfig | None = None, ):
        super().__init__()
        if config is None:
            config = MyDiffusionConfig()  # Here we use the default config
        self.config = config

        self._queues = None

        self.diffusion = MyDiffusionModel(config)

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Do something with the input batch
        # input shape: (36864, 96) = (64*2*3*96, 96)
        x = batch["observation.image"]
        y = self.model(x)

        # Return the output dictionary
        return {"loss": y}


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
            self.image_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += config.diffusion_step_embed_dim  # ?

        # U-Net
        self.unet = DiffusionUnet(config, global_cond_dim)

        
        

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    

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
            pretrained=config.pretrained_vision_backbone
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
    # config = MyDiffusionConfig()
    policy = MyDiffusionPolicy()
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