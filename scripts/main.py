'''
Main script.

Example
-------
python scripts/main.py fit --config config/mnist_conv.yaml

'''

from lightning.pytorch.cli import LightningCLI


def main():
    cli = LightningCLI()


if __name__ == '__main__':
    main()
