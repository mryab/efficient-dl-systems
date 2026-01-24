import argparse
import torch


def create_model(in_size: int, out_size: int, num_layers: int) -> torch.nn.Module:
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record CUDA memory snapshot during training")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable automatic mixed precision (FP16)")
    parser.add_argument("--in_size", type=int, default=8192, help="Input dimension of the model")
    parser.add_argument("--out_size", type=int, default=8192, help="Output dimension of the model")
    parser.add_argument("--num_layers", type=int, default=20, help="Number of linear layers in the model")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of training batches to run")
    args = parser.parse_args()

    torch.cuda.memory._record_memory_history()

    device = torch.device("cuda:0")
    data = [torch.randn(1024, args.in_size, device=device) for _ in range(args.num_batches)]
    targets = [torch.randn(1024, args.out_size, device=device) for _ in range(args.num_batches)]

    model = create_model(args.in_size, args.out_size, args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss().to(device)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    for inputs, target in zip(data, targets):
        if args.amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            output = model(inputs)
            loss = loss_fn(output, target)

            loss.backward()
            opt.step()

        opt.zero_grad()

    torch.cuda.memory._dump_snapshot(f"snapshot_amp={args.amp}.pickle")
