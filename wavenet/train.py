import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from wavenet import WaveNet
from utils import save_checkpoint, generate_audio, save_audio_batch
from mnist_audio import MNISTAudio
from config import Option

if __name__ == "__main__":
    opt = Option()
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    dataset = MNISTAudio(opt.dataset_path, opt.src_len, opt.tgt_len, opt.num_class)
    dataloader = DataLoader(dataset, batch_size=opt.batch_sz, shuffle=True, num_workers=0)

    MODEL = WaveNet(
        num_block=opt.num_block,
        num_layer=opt.num_layer,
        class_dim=opt.num_class,
        residual_dim=opt.residual_dim,
        dilation_dim=opt.dilation_dim,
        skip_dim=opt.skip_dim,
        kernel_size=opt.kernel_size,
        bias=opt.bias
    )

    optimizer = optim.Adam(MODEL.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    last_epoch = 0

    if opt.ckpt_path:
        ckpt = torch.load(opt.ckpt_path, map_location=opt.DEVICE)
        MODEL.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        losses = ckpt['losses']
        last_epoch = ckpt['last_epoch']

    MODEL.to(opt.DEVICE)

    for e in range(last_epoch, opt.epoch):
        MODEL.train()
        if opt.max_itr is None:
            pbar = tqdm(dataloader)
        else:
            pbar = tqdm(dataloader, total=min(opt.max_itr, len(dataloader)))

        accum_loss = 0
        update_count = 0

        for idx, batch in enumerate(pbar):
            if opt.max_itr is not None and idx >= opt.max_itr:
                break

            src, tgt = batch['src'].to(opt.DEVICE), batch['tgt'].to(opt.DEVICE)
            pred = MODEL(src)[:, :, -opt.tgt_len:]

            assert pred.shape[2] == tgt.shape[1], f"Prediction and target time mismatch: {pred.shape} vs {tgt.shape}"

            loss = loss_fn(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), opt.clip)
            optimizer.step()

            accum_loss += loss.item()
            update_count += 1

            if (idx + 1) % opt.loss_update_itr == 0 or (
                    opt.max_itr is not None and idx == opt.max_itr - 1
            ):
                avg_loss = accum_loss / update_count
                losses.append(avg_loss)
                accum_loss = 0
                update_count = 0
                pbar.set_description(f"Epoch {e + 1}/{opt.epoch} | Loss: {avg_loss:.4f}")

        scheduler.step()

        save_checkpoint(os.path.join(opt.ckpt_dir, f"{e}.pt"), MODEL, optimizer, {"epoch": e + 1, "losses": losses})

        if (e + 1) % 5 == 0:
            MODEL.eval()
            preview_audio = generate_audio(MODEL, 1, 8000, opt.num_class, receptive_field=opt.src_len)
            save_audio_batch(preview_audio, prefix=f"preview_e{e + 1}")

    plt.title("Training Loss")
    plt.plot(losses)
    plt.xlabel("Checkpoint Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()

    print("Generating final audio sample...")
    MODEL.eval()
    final_audio = generate_audio(MODEL, 1, 40000, opt.num_class, receptive_field=opt.src_len)
    save_audio_batch(final_audio, sample_rate=11025, prefix="final_gen_audio")