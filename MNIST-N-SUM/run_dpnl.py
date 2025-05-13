import multiprocessing

import dpnl_impl
from tools.timeout import Timeout
from dpnl.oracles.combination import CombinationOracle


def oracle_of_str(oracle_str: str, length):
    oracle_strs = oracle_str.split(",")
    if len(oracle_strs) == 1:
        if oracle_str == "enumeration":
            return dpnl_impl.enumeration(length)
        elif oracle_str == "basic":
            return dpnl_impl.basic(length)
        elif oracle_str == "logic":
            return dpnl_impl.logic(length)
        else:
            print(f"Error : invalid oracle name {oracle_str} !")
            assert 0
    else:
        return CombinationOracle([oracle_of_str(name, length) for name in oracle_strs])


def caching_hash_of_str(caching_hash_str: str, length: int):
    if caching_hash_str == "sat_logic_hash":
        return dpnl_impl.sat_logic_hash(length)
    else:
        return None


def run(I: dpnl_impl.MNISTInput, oracle_str, caching_hash_str):
    problem = dpnl_impl.problem(I)
    oracle = oracle_of_str(oracle_str, I.length)
    caching_hash = caching_hash_of_str(caching_hash_str, I.length)
    return problem.Proba(True, oracle, caching_hash)


def subprocess_run(queue: multiprocessing.Queue, I: dpnl_impl.MNISTInput, oracle_str, caching_hash_str):
    queue.put(run(I, oracle_str, caching_hash_str))


def timeout_run(timeout: float, I: dpnl_impl.MNISTInput, oracle_str, caching_hash_str):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=subprocess_run, args=(queue, I, oracle_str, caching_hash_str))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return Timeout(timeout)
    else:
        return queue.get()


if __name__ == '__main__':
    length = 4
    result = 10
    I = dpnl_impl.MNISTInput(length, result)
    print(run(I, "basic", None))
