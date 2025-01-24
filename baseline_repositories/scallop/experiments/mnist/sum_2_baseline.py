import os
import random
from typing import *
from time import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.5,), (0.5,)  # changed that
    )
])


def calculate_stats(values):
    max_value = np.max(values)
    min_value = np.min(values)
    avg_value = np.mean(values)
    std_dev = np.std(values)
    return max_value, min_value, avg_value, std_dev


class TimeLogger:
    def __init__(self):
        self.avg_iteration_times = []

    def add(self, t):
        self.avg_iteration_times.append(t)


class MNISTSumNDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            n_digits: int = 1,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.MNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)  # todo I don't think we need this
        self.n = n_digits

    def __len__(self):
        return len(self.mnist_dataset) // (2 * self.n)

    def __getitem__(self, idx):
        images = []
        digits = []
        for i in range(2 * self.n):
            img, digit = self.mnist_dataset[self.index_map[idx * 2 * self.n + i]]
            images.append(img)
            digits.append(digit)

        # Form the two numbers by concatenating digits
        first_number = int("".join(map(str, digits[:self.n])))
        second_number = int("".join(map(str, digits[self.n:])))
        label = first_number + second_number

        return images, label

    @staticmethod
    def collate_fn(batch):
        batch_images = [
            torch.stack([item[0][i] for item in batch]) for i in range(len(batch[0][0]))
        ]
        labels = torch.tensor([item[1] for item in batch]).long()

        return batch_images, labels


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test, n_digits: int):
    train_loader = torch.utils.data.DataLoader(
        MNISTSumNDataset(
            data_dir,
            n_digits=n_digits,
            train=True,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSumNDataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTSumNDataset(
            data_dir,
            train=False,
            n_digits=n_digits,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSumNDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=False
    )

    return train_loader, test_loader


class MNISTNet(nn.Module):
    def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
        super(MNISTNet, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class MNISTSumNNet(nn.Module):
    def __init__(self, n, provenance, k):
        super(MNISTSumNNet, self).__init__()
        self.n = n

        # MNIST Digit Recognition Network
        self.mnist_net = MNISTNet()
        self.reasoning_time_per_sample = []

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)

        # Add relations for each digit (2 * N relations)
        for i in range(2 * self.n):
            self.scl_ctx.add_relation(f"digit_{i + 1}", int, input_mapping=list(range(10)))

        # TODO write this as function instead of hardcoded
        # N=1
        if n == 1:
            self.scl_ctx.add_rule("summand_one(a) :- digit_1(a)")
            self.scl_ctx.add_rule("summand_two(a) :- digit_2(a)")
            max_sum_value = 9 + 9

        # N=2
        elif n == 2:
            self.scl_ctx.add_rule("summand_one(10 * a + b) :- digit_1(a), digit_2(b)")
            self.scl_ctx.add_rule("summand_two(10 * a + b) :- digit_3(a), digit_4(b)")
            max_sum_value = 99 + 99

        # N=3
        elif n == 3:
            self.scl_ctx.add_rule("summand_one(10 * 10 * a + 10 * b + c) :- digit_1(a), digit_2(b), digit_3(c)")
            self.scl_ctx.add_rule("summand_two(10 * 10 * a + 10 * b + c) :- digit_4(a), digit_5(b), digit_6(c)")
            max_sum_value = 999 + 999

        # N=4
        elif n == 4:
            self.scl_ctx.add_rule(
                "summand_one(10 * 10 * 10 * a + 10 * 10 * b + 10 * c + d) :- digit_1(a), digit_2(b), digit_3(c), digit_4(d)")
            self.scl_ctx.add_rule(
                "summand_two(10 * 10 * 10 * a + 10 * 10 * b + 10 * c + d) :- digit_5(a), digit_6(b), digit_7(c), digit_8(d)")
            max_sum_value = 9999 + 9999

        else:
            raise NotImplementedError(f"Only implemented for N = [1,2,3,4], you chose {n}")

        self.scl_ctx.add_rule("sum_2(a + b) :- summand_one(a), summand_two(b)")
        self.sum_2 = self.scl_ctx.forward_function("sum_2", output_mapping=[(i,) for i in range(0, max_sum_value)],
                                                   jit=False,
                                                   dispatch="parallel")
        print("program successfully defined")

    def forward(self, x: List[torch.Tensor]):

        digit_distrs = [self.mnist_net(imgs) for imgs in
                        x]  # List of Tensors (Batch x 10) TODO should only have two images here

        # Prepare arguments for reasoning module
        reasoning_args = {f"digit_{i + 1}": digit_distrs[i] for i in range(len(digit_distrs))}

        # Perform reasoning
        start = time()
        output = self.sum_2(**reasoning_args)
        end = time()

        reasoning_time_per_sample = (end - start) / len(x[0])
        return output, reasoning_time_per_sample


def bce_loss(output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    return F.nll_loss(output, ground_truth)


class Trainer():
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance, n_digits):
        self.model_dir = model_dir
        self.network = MNISTSumNNet(provenance=provenance, k=k, n=n_digits)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.reasoning_times = []
        self.best_loss = 10000000000
        if loss == "nll":
            self.loss = nll_loss
        elif loss == "bce":
            self.loss = bce_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")
        self.time_logger = TimeLogger()

    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        times_per_sample = []
        for (data, target) in iter:
            self.optimizer.zero_grad()
            output, avg_time = self.network(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
        times_per_sample.append(avg_time)
        return times_per_sample

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for (data, target) in iter:
                output, _ = self.network(data)
                test_loss += self.loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                perc = 100. * correct / num_items
                accuracy = correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {accuracy} ({perc:.2f}%)")
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                torch.save(self.network, os.path.join(model_dir, "sum_2_best.pt"))
        return accuracy

    def train(self, n_epochs):
        # self.test_epoch(0)
        times_per_sample = []
        for epoch in range(1, n_epochs + 1):
            times_per_sample = self.train_epoch(epoch)
            test_accuracy = self.test_epoch(epoch)
            times_per_sample += times_per_sample
        return test_accuracy, sum(times_per_sample) / len(times_per_sample)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("N", help="Number of digits per summand. Choose integers in range [1,4].", type=int, default=2,
                        nargs="?")
    parser.add_argument("exact", help="Exact reasoning or approximate reasoning (with k=3)", type=bool, default=False,
                        nargs="?")
    args = parser.parse_args()

    N = args.N

    if args.exact:
        k = 2 ** (20 * N)
    else:
        k = 3

    n_epochs = 1
    batch_size_train = 2
    batch_size_test = 64
    learning_rate = 0.001
    loss_fn = "bce"
    provenance = "difftopkproofs"
    seeds = [1, 12, 123, 1234, 12345]
    test_accuracies = []
    avg_runtimes = []
    total_times = []

    # Get the current date and time
    print("Current time:", datetime.now().strftime("%H:%M"))

    for i, seed in enumerate(seeds):
        total_start = time()
        torch.manual_seed(seed)
        random.seed(seed)

        # Data
        data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
        model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_sum_2"))
        os.makedirs(model_dir, exist_ok=True)

        # Dataloaders
        train_loader, test_loader = mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test,
                                                       n_digits=N)  # todo adapt to the number of images per task

        # Create trainer and train
        trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance,
                          n_digits=N)  # todo adapt to the number of images per task
        test_accuracy, avg_runtimes_per_iter = trainer.train(n_epochs)

        print(f'Run {i} with seed {seed}: test_accuracy={test_accuracy}, avg_reasoning_time={avg_runtimes_per_iter}')
        test_accuracies.append(test_accuracy)
        avg_runtimes.append(avg_runtimes_per_iter)
        total_times.append(time() - total_start)

    accuracy_stats = calculate_stats(test_accuracies)
    runtime_stats = calculate_stats(avg_runtimes)
    num_runs = len(test_accuracies)

    output_str = f"Parameters: n_epochs = {n_epochs}, batch_size={batch_size_train}, lr={learning_rate}, k={k}, num_runs={len(seeds)}, provenance={provenance}, N_digits={N}\n"
    output_str += "Test Accuracy and Runtime for Each Run:\n"
    for i, seed in enumerate(seeds):
        output_str += f"Run {i + 1} (seed: {seed}) - Test Accuracy: {test_accuracies[i]:.4f}, Avg Runtime: {avg_runtimes[i]:.4f}s, Total time: {total_times[i]:.4f}s\n"
    output_str += "\nSummary Statistics:\n"
    output_str += f"Test Accuracy - Max: {accuracy_stats[0]:.4f}, Min: {accuracy_stats[1]:.4f}, Avg: {accuracy_stats[2]:.4f}, Std Dev: {accuracy_stats[3]:.4f}\n"
    output_str += f"Runtime - Max: {runtime_stats[0]:.4f} seconds, Min: {runtime_stats[1]:.4f} seconds, Avg: {runtime_stats[2]:.4f} seconds, Std Dev: {runtime_stats[3]:.4f} seconds\n"

    print(output_str)
    file_name = f"scallop_summary_N{N}_exact.txt"
    with open(file_name, "w") as file:
        file.write(output_str)
