'''VAE training on MNIST.'''

from lightning.pytorch.cli import LightningCLI

from varautoenc import BinarizedMNIST


def main():
    cli = LightningCLI(datamodule_class=BinarizedMNIST)


if __name__ == '__main__':
    main()

