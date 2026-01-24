import argparse

import torch

from tqdm.auto import tqdm

from mnist_training import get_loaders
from train import create_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transforms-level", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    train_dataloader_, val_dataloader_ = get_loaders(args.transforms_level)
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch = 0
    i = 0
    model.to(device)
    model.train()

    with torch.autograd.profiler.emit_nvtx():
        for x_train, y_train in tqdm(train_dataloader_, desc=f"Epoch {epoch}: "):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                x_train, y_train = x_train.to(device), y_train.to(device)
                y_pred = model(x_train)
                loss = loss_fn(y_pred, y_train)

            # you can add your own event markers with range_push()
            # and mark the end of the event with range_pop()
            torch.cuda.nvtx.range_push("[now you see me] Backward")
            scaler.scale(loss).backward()
            torch.cuda.nvtx.range_pop()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            i += 1
            if i == 3:
                break
