# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""MNIST example of Transformer Engine Paddle"""

import argparse
import unittest

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddle.vision.transforms import Normalize
from paddle.io import DataLoader
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy

import transformer_engine.paddle as te


class Net(nn.Layer):
    """Simple network used to train on MNIST"""

    def __init__(self, use_te=False):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 32, 3, 1)
        self.conv2 = nn.Conv2D(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        if use_te:
            self.fc1 = te.Linear(9216, 128)
            self.fc2 = te.Linear(128, 16)
        else:
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        """FWD"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = paddle.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(args, model, train_loader, optimizer, epoch):
    """Training function."""
    model.train()
    for batch_id, (data, labels) in enumerate(train_loader):
        with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()

        if batch_id % args.log_interval == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_id * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_id / len(train_loader):.0f}%)]\t"
                  f"Loss: {loss.item():.6f}")
            if args.dry_run:
                return loss.item()
    return loss.item()


def evaluate(model, test_loader, epoch):
    """Testing function."""
    model.eval()
    metric = Accuracy()
    metric.reset()

    with paddle.no_grad():
        for data, labels in test_loader:
            with paddle.amp.auto_cast(dtype='bfloat16', level='O2'):    # pylint: disable=not-context-manager
                outputs = model(data)
                acc = metric.compute(outputs, labels)
            metric.update(acc)
    print(f"Epoch[{epoch}] - accuracy: {metric.accumulate():.6f}")
    return metric.accumulate()


def mnist_parser(args):
    """Parse training settings"""
    parser = argparse.ArgumentParser(description="Paddle MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--use-te",
                        action="store_true",
                        default=False,
                        help="Use Transformer Engine")
    return parser.parse_args(args)


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)

    paddle.seed(args.seed)

    # Load MNIST dataset
    transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
    train_dataset = MNIST(mode='train', transform=transform)
    val_dataset = MNIST(mode='test', transform=transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size)

    # Define model and optimizer
    model = Net(use_te=args.use_te)
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())

    # Cast model to BF16
    model = paddle.amp.decorate(models=model, level='O2', dtype='bfloat16')

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, train_loader, optimizer, epoch)
        acc = evaluate(model, val_loader, epoch)

    return loss, acc


class TestMNIST(unittest.TestCase):
    """MNIST unittests"""

    @classmethod
    def setUpClass(cls):
        """Run MNIST without Transformer Engine"""
        cls.args = mnist_parser(["--epochs", "5"])

    @staticmethod
    def verify(actual):
        """Check If loss and accuracy match target"""
        desired_traing_loss = 0.5
        desired_test_accuracy = 0.98
        assert actual[0] < desired_traing_loss
        assert actual[1] > desired_test_accuracy

    @unittest.skipIf(paddle.device.cuda.get_device_capability() < (8, 0),
                     "BF16 MNIST example requires Ampere+ GPU")
    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        self.args.use_te = True
        actual = train_and_evaluate(self.args)
        self.verify(actual)


if __name__ == "__main__":
    train_and_evaluate(mnist_parser(None))