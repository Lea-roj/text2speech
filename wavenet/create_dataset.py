import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from utils import get_files, quantize


def create_dataset(dataset_path, files, sr=11025, bit_depth=8):
    dataset = []
    print(f"Creating dataset with {len(files)} files...")
    for f in tqdm(files, desc="Processing audio files"):
        try:
            wav, _ = librosa.load(f, sr=sr)
            quantized_wav = quantize(wav, bit_depth)
            quantized_wav = np.clip(quantized_wav, 0, 2 ** bit_depth - 1)
            dataset.append(quantized_wav)
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