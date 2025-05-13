from typing import Any
import neural

from dpnl.core.symbolic import Input, Symbolic
from dpnl.core.variable import RndVar, unknown
from dpnl.core.problem import PNLProblem
from dpnl.oracles.enumeration import EnumerationOracle
from dpnl.oracles.basic import BasicOracle
from dpnl.oracles.logic import LogicOracle
from dpnl.logics.sat_logic import SATLogic, SATLogicSymbolic
from dpnl.cachings.sat_logic_hash import SATCachingHash


# Neural integration into DPNL
class MNISTDigitVar(RndVar):
    def __init__(self, name, network: neural.MNISTNetwork, img: neural.MNISTImage):
        self.network = network
        self.img = img
        tensor = network.forward(img.tensor)
        super().__init__(name, {digit: tensor[0][digit] for digit in range(10)})

    def update(self, img: neural.MNISTImage = None):
        if img is not None:
            self.img = img
        tensor = self.network.forward(self.img.tensor)
        self.domain_distrib = {digit: tensor[0][digit] for digit in range(10)}


class MNISTInput(Input):
    def __init__(self, network: neural.MNISTNetwork, n1_images: list, n2_images: list):
        assert len(n1_images) == len(n2_images)
        self.length = len(n1_images)
        n1 = int(''.join(str(img.label) for img in n1_images))
        n2 = int(''.join(str(img.label) for img in n2_images))
        self.result = n1 + n2
        self.num = [
            [MNISTDigitVar(("num", 0, idx), network, n1_images[idx]) for idx in range(self.length)],
            [MNISTDigitVar(("num", 0, idx), network, n2_images[idx]) for idx in range(self.length)]
        ]
        super().__init__(probabilistic_attributes={"num"})

    def update(self):
        for var in self:
            var.update()


class MNISTSymbolic(Symbolic):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __call__(self, I: MNISTInput):
        carry = 0
        r = I.result
        for i in range(I.length):
            idx = I.length - i - 1
            d = I.num[0][idx].value + I.num[1][idx].value + carry
            if d % 10 != r % 10:
                return False
            r = int(r / 10)
            carry = int(d / 10)
        return carry == r


# Creating datasets

def dataset(network: neural.MNISTNetwork, length: int, dataset_type: str):
    img_list = neural.datasets[dataset_type]
    size = len(img_list) // (2 * length)
    data = []
    for idx in range(size):
        n1_idx = idx * 2 * length
        n2_idx = n1_idx + length
        n1_images = [img_list[n1_idx + i] for i in range(length)]
        n2_images = [img_list[n2_idx + i] for i in range(length)]
        data.append(MNISTInput(network, n1_images, n2_images))
    return data