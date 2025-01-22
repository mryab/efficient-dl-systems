import pytest
import torch

from modeling.diffusion import DiffusionModel
from modeling.unet import UnetModel


@pytest.mark.parametrize(
    [
        "input_tensor",
        "num_timesteps",
    ],
    [
        (
            torch.randn(2, 3, 32, 32),
            10,
        ),
        (
            torch.randn(2, 3, 64, 64),
            20,
        ),
        (
            torch.randn(2, 3, 128, 128),
            30,
        ),
        (
            torch.randn(2, 3, 256, 256),
            40,
        ),
    ],
)
def test_unet(input_tensor, num_timesteps):
    B, C, H, W = input_tensor.shape
    net = UnetModel(C, C, hidden_size=128)
    timestep = torch.randint(1, num_timesteps + 1, (B,)) / num_timesteps
    out = net(input_tensor, timestep)
    assert out.shape == input_tensor.shape


def test_diffusion(num_channels=3, batch_size=4):
    # note: you should not need to change the thresholds or the hyperparameters
    net = UnetModel(num_channels, num_channels, hidden_size=128)
    model = DiffusionModel(eps_model=net, betas=(1e-4, 0.02), num_timesteps=20)

    input_data = torch.randn((batch_size, num_channels, 32, 32))

    output = model(input_data)
    assert output.ndim == 0
    assert 1.0 <= output <= 1.2
