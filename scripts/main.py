'''VAE training on MNIST.'''

from lightning.pytorch.cli import LightningCLI

from varautoenc import MNISTDataModule


def main():
    cli = LightningCLI(datamodule_class=MNISTDataModule)


if __name__ == '__main__':
    main()

