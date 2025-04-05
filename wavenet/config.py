from dataclasses import dataclass


@dataclass
class Option:
    DEVICE: str = 'cpu'

    # Training
    epoch: int = 100
    lr: float = 1e-3
    batch_sz: int = 4
    clip: float = 1.0
    max_itr: int = 500
    loss_update_itr: int = 20

    # Model
    num_class: int = 256
    src_len: int = 1088
    tgt_len: int = 64
    num_block: int = 4
    num_layer: int = 10
    residual_dim: int = 128
    dilation_dim: int = 128
    skip_dim: int = 256
    kernel_size: int = 2
    bias: bool = False

    dataset_path: str = "wavenet_dataset.npz"
    ckpt_path: str = ""
    ckpt_dir: str = "ckpt"
