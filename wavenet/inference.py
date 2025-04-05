import os
import torch
import matplotlib.pyplot as plt
from wavenet import WaveNet
from utils import generate_audio, save_audio_batch
from config import Option

if __name__ == "__main__":
    opt = Option()
    opt.DEVICE = "cpu"
    checkpoint_path = os.path.join(opt.ckpt_dir, "8.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    model = WaveNet(
        num_block=opt.num_block,
        num_layer=opt.num_layer,
        class_dim=opt.num_class,
        residual_dim=opt.residual_dim,
        dilation_dim=opt.dilation_dim,
        skip_dim=opt.skip_dim,
        kernel_size=opt.kernel_size,
        bias=opt.bias
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=opt.DEVICE)["model_state_dict"])
    model.to(opt.DEVICE)
    model.eval()

    print("Generating audio...")
    generated_audio = generate_audio(
        model=model,
        audio_num=1,
        audio_len=110250,       # samples
        num_class=opt.num_class,
        receptive_field=opt.src_len
    )

    save_audio_batch(generated_audio, sample_rate=11025, prefix="gen_audio")

    plt.title("Generated Audio Waveform")
    plt.plot(generated_audio[0])
    plt.savefig("generated_waveform.png")
    plt.show()
