from json import dumps
from datetime import datetime
import time
import torch
import random
import numpy as np
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

method = "geometric_mean"
N_values = [4]
batch_size = 2
learning_rate = 1e-3
timestamp = datetime.now().strftime("%m-%d-%H-%M")
# seeds = [0, 1]
# seeds = [1, 12, 123, 1234, 12345]
seeds = [0]
test_accuracies, symbolic_iter_times = [], []

for i in N_values:
    for n, _seed in enumerate(range(len(seeds))):
        print(f"\n#################### STARTING RUN {n} WITH SEED {seeds[n]} ####################\n")
        total_start = time.time()
        torch.manual_seed(_seed)
        np.random.seed(_seed)
        name = "results_seed{}".format(_seed)

        train_set = addition(i, "train")
        test_set = addition(i, "test")

        network = MNIST_Net()

        pretrain = 0
        if pretrain is not None and pretrain > 0:
            network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
        net = Network(network, "mnist_net", batching=True)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)  # changed, previous was 1e-3

        model = Model("models/addition.pl", [net])
        if method == "exact":
            model.set_engine(ExactEngine(model), cache=True)
        elif method == "geometric_mean":
            model.set_engine(
                ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
            )

        model.add_tensor_source("train", MNIST_train)
        model.add_tensor_source("test", MNIST_test)

        loader = DataLoader(train_set, batch_size, False)
        train = train_model(model, loader, 1, log_iter=100, profile=0)
        model.save_state("snapshot/" + name + ".pth")
        train.logger.comment(dumps(model.get_hyperparameters()))

        test_accuracy = get_confusion_matrix(model, test_set, verbose=1).accuracy()
        train.logger.comment(
            "Accuracy {}".format(test_accuracy)
        )
        test_accuracies.append(test_accuracy.item())
        symbolic_iter_times += list(train.logger.log_data['total_iter_time'].values())
        train.logger.write_to_file("log/" + f"{method}/" + f"{i}Sum/" + name)
        print(f" total time: {time.time() - total_start}")

    # Initialize the string that will hold all the results
    result_str = "=" * 70 + "\n"
    result_str += "Test Accuracies for Each Run\n"
    result_str += "=" * 70 + "\n"

    # Loop over each run and append the results to the string
    for i, accuracy in enumerate(test_accuracies, start=1):
        result_str += f"Run {i}: Test Accuracy: {accuracy:.4f}\n"

    # Calculate the summary statistics
    mean_accuracy = np.mean(test_accuracies)
    max_accuracy = np.max(test_accuracies)
    min_accuracy = np.min(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    mean_symbolic_iter_time = np.mean(symbolic_iter_times)

    # Append the summary statistics to the string
    result_str += "=" * 70 + "\n"
    result_str += "Summary Statistics\n"
    result_str += "=" * 70 + "\n"
    result_str += f"Mean Test Accuracy: {mean_accuracy:.6f}\n"
    result_str += f"Max Test Accuracy: {max_accuracy:.6f}\n"
    result_str += f"Min Test Accuracy: {min_accuracy:.6f}\n"
    result_str += f"Standard Deviation of Test Accuracy: {std_accuracy:.6f}\n"
    result_str += f"Mean Symbolic Iteration time: {mean_symbolic_iter_time:.6f}s\n"

    # Print the result string
    print(result_str)

    # Write the result string to a file
    with open("log/" + f"{method}/" + f"{i}Sum/" + "summary_timeout.txt", 'w') as file:
        file.write(result_str)
