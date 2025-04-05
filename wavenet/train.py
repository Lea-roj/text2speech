import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from wavenet import WaveNet
from utils import save_ckpt, generate_audio
from mnist_audio import MNISTAudio
from config import Option

if __name__ == "__main__":
    opt = Option()
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    dataset = MNISTAudio(opt.dataset_path, opt.src_len, opt.tgt_len, opt.num_class)
    dataloader = DataLoader(dataset, batch_size=opt.batch_sz, shuffle=True, num_workers=0)
    pbar = tqdm(range(opt.epoch * min(opt.max_itr, len(dataloader))))

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
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    last_epoch = 0

    if opt.ckpt_path:
        ckpt = torch.load(opt.ckpt_path)
        MODEL.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        losses = ckpt['losses']
        last_epoch = ckpt['last_epoch']

    for e in range(last_epoch, opt.epoch):
        MODEL.train()
        MODEL.to(opt.DEVICE)
        accum_loss = 0

        for idx, batch in enumerate(dataloader):
            src, tgt = batch['src'].to(opt.DEVICE), batch['tgt'].to(opt.DEVICE)
            pred = MODEL(src)[:, :, -opt.tgt_len:]
            loss = loss_fn(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), opt.clip)
            optimizer.step()

            accum_loss += loss.item()
            pbar.update()

            if idx == opt.max_itr:
                break

            if (idx + 1) % opt.loss_update_itr == 0:
                avg_loss = accum_loss / opt.loss_update_itr
                losses.append(avg_loss)
                accum_loss = 0
                pbar.set_description(f"Epoch {e} | Loss: {avg_loss:.4f}")

        save_ckpt(os.path.join(opt.ckpt_dir, f"{e}.pt"), MODEL, optimizer, {"epoch": e + 1, "losses": losses})

    plt.title("Training Loss")
    plt.plot(losses)
    plt.savefig("loss_plot.png")

    MODEL.to(opt.DEVICE)
    audio = generate_audio(MODEL, 1, 40000, opt.num_class)