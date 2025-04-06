import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.io.wavfile import write


def mu_law_companding_transformation(x, mu=255):
    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))


def inverse_mu_law_companding_transformation(y, mu=255):
    return np.sign(y) * (((1 + mu) ** np.abs(y) - 1) / mu)


def quantize(wav, bit):
    wav = mu_law_companding_transformation(wav, 2**bit - 1)
    return ((wav + 1) * 2**(bit - 1)).astype(int)


def inv_quantize(wav, bit):
    mu = 2**bit - 1
    wav = wav.astype(np.float32)
    wav = (wav / mu) * 2 - 1
    return inverse_mu_law_companding_transformation(wav, mu)


def get_files(directory, ext=".wav"):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_paths.append(os.path.join(root, file))
    return sorted(file_paths)


def save_checkpoint(path, model, optimizer, misc):
    model.eval()
    model.cpu()
    torch.save({
        'last_epoch': misc.get('epoch', 0),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': misc.get('losses', []),
        'config': misc.get('config', None)
    }, path)


def generate_audio(model, audio_num, audio_len, num_class, receptive_field=1024, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    # Pad initial input with zeros to match receptive field
    input_ = torch.zeros(audio_num, num_class, receptive_field, device=device)
    samples = []

    for _ in tqdm(range(audio_len), desc="Generating audio"):
        # Predict next distribution using most recent context
        pred = model(input_[:, :, -receptive_field:])
        logits = pred[:, :, -1] / max(temperature, 1e-5)
        dist = F.softmax(logits, dim=1)

        # Sample from distribution
        sample = torch.multinomial(dist, num_samples=1)
        samples.append(sample)

        # Convert to one-hot and append to input sequence
        one_hot = F.one_hot(sample.view(audio_num, 1), num_class).float().permute(0, 2, 1)
        input_ = torch.cat((input_, one_hot), dim=-1)

    # Concatenate sampled indices into a waveform
    output = torch.cat(samples, dim=1).squeeze(0).cpu().numpy()
    output = inv_quantize(output, bit=8)

    # Optional: clip to safe range
    output = np.clip(output, -1.0, 1.0)
    return [output]


def save_audio_batch(audio_batch, sample_rate=11025, prefix="output"):
    for i, audio in enumerate(audio_batch):
        # Ensure float32 and normalize range
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)              # avoid overflow
        audio = (audio * 32767).astype(np.int16)       # convert to 16-bit PCM

        path = f"{prefix}_{i}.wav"
        write(path, sample_rate, audio)
        print(f"Saved: {path} ({len(audio) / sample_rate:.2f} sec)")
