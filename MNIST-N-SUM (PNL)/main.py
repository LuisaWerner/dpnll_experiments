import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam

from neural import MNISTNetwork
from dpnl_impl import dataset, MNISTSymbolic

from dpnl.core.problem import PNLProblem
from dpnl.oracles.basic import BasicOracle


class Logger:
    def __init__(self, column_width: int = 40, avg_log_freq: int = 100):
        self.avg_log_freq = avg_log_freq
        self.column_width = column_width
        self.logs = []

    def _print(self, log: list[tuple[str, Any]]):
        tmp = ""
        for key, value in log:
            if isinstance(value, float):
                value = f"{value:.4e}"
            unit = f"\t{key} = {value}\t"
            tmp += f"{unit:<{self.column_width}}|"
        print(tmp)

    def log(self, log: list[tuple[str, Any]]):
        self.logs.append(log)
        if len(self.logs) > self.avg_log_freq and len(self.logs) % self.avg_log_freq == 0:
            self.print_last_avg(self.avg_log_freq)

    def print_last(self):
        self._print(self.logs[len(self.logs) - 1])

    def print_last_avg(self, number: int):
        count = {}
        sums = {}
        for idx in range(1, number + 1):
            log = self.logs[-idx]
            for key, value in log:
                count[key] = count.get(key, 0) + 1
                sums[key] = sums.get(key, 0.0) + value
        report = [("Iters", number)] + [(f"Avg {key}", float(sums[key]) / count[key]) for key in count]
        self._print(report)


def train_digit_classifier(network: MNISTNetwork, length: int, epochs: int = 1, lr: float = 1e-4,
                           logger: Logger = None):
    if logger is None:
        logger = Logger()

    train_set = dataset(network, length, "train")
    bce_loss = nn.BCELoss()
    optimizer = Adam(network.parameters(), lr=lr)

    pnl_problem = PNLProblem(train_set[0], MNISTSymbolic(length))
    oracle = BasicOracle(pnl_problem.S)

    network.train()
    for epoch in range(epochs):
        total_loss = 0
        for mnist_input in train_set:
            t = time.time()
            mnist_input.update()
            pnl_problem.I = mnist_input
            estimation = pnl_problem.Proba(True, oracle)
            loss = bce_loss(estimation, torch.tensor(1.0))
            loss.backward()
            optimizer.step()
            t = time.time() - t
            logger.log([("Iteration Time (s.)", t), ("Loss", loss.item())])
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    print("Digit classifier training completed.")


def accuracy(network: MNISTNetwork, length: int):
    test_set = dataset(network, length, "test")
    pnl_problem = PNLProblem(test_set[0], MNISTSymbolic(length))
    oracle = BasicOracle(pnl_problem.S)
    count = 0
    for mnist_input in test_set:
        pnl_problem.I = mnist_input
        estimation = pnl_problem.Proba(True, oracle)
        if estimation.item() >= 0.5:
            count += 1
    return float(count) / len(test_set)


if __name__ == "__main__":
    length = 2
    net = MNISTNetwork()
    logger = Logger(column_width=55, avg_log_freq=5)
    train_digit_classifier(net, 4, logger=logger)
    logger.log([("Accuracy", accuracy(net, length))])
    logger.print_last()
