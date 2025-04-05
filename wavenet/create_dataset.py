import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_files, quantize, mu_law_companding_transformation, inverse_mu_law_companding_transformation


def create_dataset(dataset_path, files, sr=11025, bit_depth=8):
    dataset = []
    mu = 2 ** bit_depth - 1
    print(f"Creating dataset with {len(files)} files...")

    for i, f in enumerate(tqdm(files, desc="Processing audio files")):
        try:
            wav, _ = librosa.load(f, sr=sr)
            wav = np.clip(wav, -1.0, 1.0)
            quantized_wav = quantize(wav, bit_depth)
            quantized_wav = np.clip(quantized_wav, 0, mu)
            dataset.append(quantized_wav)

            if i == 0:
                mu_law_wav = mu_law_companding_transformation(wav, mu=mu)
                reconstructed_wav = inverse_mu_law_companding_transformation(mu_law_wav, mu=mu)

                plt.figure(figsize=(14, 8))

                plt.subplot(3, 1, 1)
                plt.plot(wav[:1000])
                plt.title("Original Audio (first 1000 samples)")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.grid(True)

                plt.subplot(3, 1, 2)
                plt.plot(mu_law_wav[:1000], color='orange')
                plt.title("Mu-law Companded Audio (first 1000 samples)")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.grid(True)

                plt.subplot(3, 1, 3)
                plt.plot(reconstructed_wav[:1000], color='green')
                plt.title("Reconstructed Audio (after inverse mu-law, first 1000 samples)")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.grid(True)

                plt.tight_layout()
                os.makedirs("figures", exist_ok=True)
                plot_path = "figures/mu_law_roundtrip_comparison.png"
                plt.savefig(plot_path, dpi=300)
                print(f"Saved mu-law roundtrip comparison figure to {plot_path}")

        except Exception as e:
            print(f"[Warning] Failed to load {f}: {e}")

    np.savez(dataset_path, *dataset)
    print(f"Dataset saved to: {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and create .npz dataset for WaveNet training.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to folder containing .wav files")
    parser.add_argument("--out_path", type=str, default="dataset.npz", help="Output path for dataset file")
    parser.add_argument("--sr", type=int, default=11025, help="Target sampling rate")
    parser.add_argument("--bit", type=int, default=8, help="Bit depth for quantization (mu-law)")

    args = parser.parse_args()

    files = get_files(args.audio_dir)
    if not files:
        print("No audio files found")
        exit(1)

    create_dataset(args.out_path, files, sr=args.sr, bit_depth=args.bit)