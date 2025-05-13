import time
import tools
import problog_impl
import dpnl_impl
from run_dpnl import timeout_run


def run_dpnl(I, oracle_str, timeout):
    start_time = time.time()
    result = timeout_run(timeout, I, oracle_str, "")
    duration = time.time() - start_time
    return result, duration


def run_problog(I, timeout):
    start_time = time.time()
    result = tools.problog_interface.problog_run(problog_impl.new_program(I), timeout)
    duration = time.time() - start_time
    return result, duration


def format_result(result, width=40):
    """Format the result and duration for aligned output."""
    result, duration = result
    if isinstance(result, float):
        formatted = f"{result:.4e} ({duration:.2f}s)"
    elif isinstance(result, dict):
        for value in result.values():
            result = value
            break
        formatted = f"{result:.4e} ({duration:.2f}s)"
    else:
        formatted = repr(result)
    return f"{formatted:<{width}}"


def main():
    timeout = 120
    column_width = 40

    headers = [
        "N",
        "DPNL Basic Oracle",
        "DPNL Logic Oracle",
        "DPNL Basic+Logic Oracle",
        "DPNL Hand-Crafted Oracle",
        "ProbLog"
    ]

    print("This program will last for around 30min...")
    print("\n=== Comparison: DPNL vs. ProbLog on Graph Reachability ===")
    print("Each row corresponds to a random NxN graph (N = number of nodes).")
    print("Each possible edge have a certain probability of being present in the graph.")
    print("We estimate the probability that there is a path from node 0 to node (N-1).")
    print("Columns show the probability result and time taken (in seconds):")
    print(" - DPNL Basic Oracle        = Inference using the automatically-generated basic oracle")
    print(" - DPNL Logic Oracle        = Inference using the complete hand-made oracle")
    print(" - DPNL Basic+Logic Oracle  = Inference using the combination of the Basic and Logic oracles")
    print(" - DPNL Hand-Crafted Oracle = Inference using the optimized complete hand-crafted oracle")
    print(" - ProbLog                  = Result from the ProbLog logic programming system")
    print("Format: <probability> (<time>s)")
    print()

    print(" | ".join(f"{header:<{column_width}}" if i else f"{header:<3}" for i, header in enumerate(headers)))
    print("-" * (column_width * (len(headers) - 1) + 5))

    for length in range(2, 10):
        I = dpnl_impl.GraphReachInput(length, 0, length-1)
        I.randomize_probabilities()

        print(
            f"{length:<3} | "
            f"{format_result(run_dpnl(I, "basic", timeout), column_width)} | "
            f"{format_result(run_dpnl(I, "logic", timeout), column_width)} | "
            f"{format_result(run_dpnl(I, "basic,logic", timeout), column_width)} | "
            f"{format_result(run_dpnl(I, "hand-crafted", timeout), column_width)} | "
            f"{format_result(run_problog(I, timeout), column_width)}"
        )


if __name__ == "__main__":
    main()
