import argparse

import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 3,
    device: torch.device = "cuda:0",
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        i = 0
        for x_train, y_train in tqdm(train_dataloader, desc=f"Epoch {epoch}: "):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_pred = model(x_train)

            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1
            if i == 10 and n_epochs == 1:
                break

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print("Starting validation...")
            model.eval()
            val_loss = torch.empty(len(val_dataloader))
            val_accuracy = torch.empty(len(val_dataloader))

            with torch.no_grad():
                for i, (x_val, y_val) in enumerate(val_dataloader):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = model(x_val)
                    loss = loss_fn(y_pred, y_val)
                    val_loss[i] = loss
                    val_accuracy[i] = (torch.argmax(y_pred, dim=-1) == y_val).float().mean()

            print(
                f"Epoch: {epoch}, loss: {val_loss.mean().detach().cpu()}, "
                f"accuracy: {val_accuracy.mean().detach().cpu()}"
            )
    model.eval()


def train_amp(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 3,
    device: torch.device = "cuda:0",
) -> None:
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        for x_train, y_train in tqdm(train_dataloader, desc=f"Epoch {epoch}: "):
            with torch.cuda.amp.autocast():
                x_train, y_train = x_train.to(device), y_train.to(device)
                y_pred = model(x_train)

                loss = loss_fn(y_pred, y_train)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print("Starting validation...")
            model.eval()
            val_loss = torch.empty(len(val_dataloader))
            val_accuracy = torch.empty(len(val_dataloader))

            with torch.no_grad():
                for i, (x_val, y_val) in enumerate(val_dataloader):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = model(x_val)
                    loss = loss_fn(y_pred, y_val)
                    val_loss[i] = loss
                    val_accuracy[i] = (torch.argmax(y_pred, dim=-1) == y_val).float().mean()

            print(
                f"Epoch: {epoch}, loss: {val_loss.mean().detach().cpu()}, "
                f"accuracy: {val_accuracy.mean().detach().cpu()}"
            )
    model.eval()


def get_loaders(transforms_level: int = 1) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if transforms_level == 1:
        # no transforms
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif transforms_level == 2:
        # modest transforms
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomPerspective(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        # heavy transforms
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomPerspective(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.GaussianBlur(5),
                torchvision.transforms.RandomAdjustSharpness(2),
                torchvision.transforms.RandomAutocontrast(),
                torchvision.transforms.RandomAdjustSharpness(1),
                torchvision.transforms.RandomAutocontrast(),
                torchvision.transforms.RandomAdjustSharpness(0.5),
                torchvision.transforms.RandomAutocontrast(),
                torchvision.transforms.RandomEqualize(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomSolarize(0.5),
                torchvision.transforms.RandomSolarize(0.5),
                torchvision.transforms.RandomSolarize(0.5),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    mnist_train = torchvision.datasets.MNIST("./mnist/", train=True, download=True, transform=transform)
    mnist_val = torchvision.datasets.MNIST("./mnist/", train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=1024, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(mnist_val, batch_size=1024, shuffle=False)

    return train_dataloader, val_dataloader


def get_model(model_level: int = 1) -> nn.Module:
    if model_level == 1:
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2 * 2 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    elif model_level == 2:
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2 * 2 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
    else:
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2 * 2 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-level", type=int, default=1)
    parser.add_argument("-t", "--transforms-level", type=int, default=1)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--n-epochs", type=int, default=100)
    args = parser.parse_args()

    train_dataloader_, val_dataloader_ = get_loaders(args.transforms_level)
    model_ = get_model(args.model_level)
    if args.amp:
        train_amp(model_, train_dataloader_, val_dataloader_, n_epochs=args.n_epochs)
    else:
        train(model_, train_dataloader_, val_dataloader_, n_epochs=args.n_epochs)
