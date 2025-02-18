import argparse
import torch
from tqdm.auto import trange

device = torch.device("cuda:0")

in_size = 8192
out_size = 8192
num_layers = 20
num_batches = 10
epochs = 1

def make_model(in_size: int, out_size: int, num_layers: int) -> torch.nn.Module:
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", action="store_true", default=False)
    args = parser.parse_args()

    torch.cuda.memory._record_memory_history()

    data = [torch.randn(1024, in_size, device=device) for _ in range(num_batches)]
    targets = [torch.randn(1024, out_size, device=device) for _ in range(num_batches)]

    net = make_model(in_size, out_size, num_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in trange(epochs):
        for inputs, target in zip(data, targets):
            if args.amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = net(inputs)
                    loss = loss_fn(output, target)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            else:
                output = net(inputs)
                loss = loss_fn(output, target)

                loss.backward()
                opt.step()
                opt.zero_grad()

    torch.cuda.memory._dump_snapshot(f"snapshot_amp={args.amp}.pickle")
