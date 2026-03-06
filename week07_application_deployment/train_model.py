import torchvision.models as models
import torch


def main():
    vgg16 = models.vgg16(pretrained=True)

    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(vgg16, example)
    traced_script_module.save("vgg16.pt")


if __name__ == "__main__":
    main()
