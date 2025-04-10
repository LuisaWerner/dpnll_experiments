import multiprocessing
import random
import time
import tempfile
import subprocess
import signal
import sys
from dpnl import (
    BoolRndVar, PNLProblem,
    BasicOracle, unknown
)
from graph_reachability import (
    graph_reachability, OptimizedGraphReachabilityOracle, random_graph
)
import z3_logic

# == Handling the kill of subprocess when the program is killed ==
active_subprocess = {}


def handle_sigterm(signum, frame):
    print("Received termination signal. Performing cleanup...")
    print("active subprocess : ", active_subprocess)
    print("Killing active subprocess...")
    for proc in active_subprocess.values():
        proc.kill()
        proc.wait()
    print("Cleanup complete. Exiting.")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)


# === DPNL Run Function ===

def run_dpnl(graph, mode):
    N = len(graph)
    pnl_problem = PNLProblem((graph, 0, 1), graph_reachability)

    if mode == "basic":
        oracle = BasicOracle(graph_reachability)
    elif mode == "complete":
        oracle = OptimizedGraphReachabilityOracle()
    elif mode == "logic":
        X = ()
        for i in range(N):
            for j in range(N):
                X += (graph[i][j],)
        S = z3_logic.graph_reachability_S(N, 0, 1)
        pnl_problem = PNLProblem(X, S)
        oracle = z3_logic.Z3Logic.LogicOracleMonotone(S)
    else:
        raise ValueError("Unknown mode")

    prob = pnl_problem.prob(oracle, True)
    return prob


# === ProbLog Run Function (via subprocess) ===

def run_problog(graph, timeout=30):
    # Generate edge facts
    facts = ""
    for i in range(N):
        for j in range(N):
            if i != j:
                if graph[i][j].value is True:
                    facts += f"edge({i},{j}).\n"
                elif not graph[i][j].defined():
                    facts += f"{graph[i][j].domain_distrib[True]:.5f}::edge({i},{j}).\n"

    # Reachability rules and query
    rules = """
    reachable(X,Y) :- edge(X,Y).
    reachable(X,Y) :- edge(X,Z), reachable(Z,Y).
    query(reachable(0,1)).
    """

    full_program = facts + rules

    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as f:
        f.write(full_program)
        filename = f.name

    try:
        start_time = time.time()

        # Launch subprocess
        proc = subprocess.Popen(
            ["problog", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        active_subprocess[id(proc)] = proc


        # Wait with timeout
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            elapsed_time = time.time() - start_time

            # Parse ProbLog output
            for line in stdout.splitlines():
                if line.startswith("reachable(0,1):"):
                    prob = float(line.split(":")[1].strip())
                    return prob, elapsed_time  # No timeout

            # No match found
            return "FAILED", elapsed_time

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            del active_subprocess[id(proc)]
            return "TIMEOUT", timeout

    except Exception as e:
        return f"ERROR: {e}", 0.0


# === Timeout Wrapper ===

def process_wrapper(queue, func, args):
    try:
        start = time.time()
        result = func(*args)
        duration = time.time() - start
        queue.put((result, duration))
    except Exception as e:
        queue.put((f"ERROR: {e}", 0))


def run_with_timeout(func, args=(), timeout=30):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=process_wrapper, args=(queue, func, args))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return "TIMEOUT", timeout

    if not queue.empty():
        return queue.get()
    return "ERROR: No result returned", 0


# === Main Test Loop ===

def format_result(result, duration, width=25):
    """Formats the result and duration to align columns."""
    if isinstance(result, float):
        result_str = f"{result:.8f} ({duration:.2f}s)"
    else:
        result_str = f"{str(result):<8} ({duration:.2f}s)"
    return f"{result_str:<{width}}"


if __name__ == "__main__":

    print("This program will last for around 20min...")
    print("\n=== 🔍 Comparison: DPNL vs. ProbLog on Graph Reachability ===")
    print("Each row corresponds to a random NxN graph (N = number of nodes).")
    print("Each possible edge have a certain probability of being present in the graph.")
    print("We estimate the probability that there is a path from node 0 to node 1.")
    print("Columns show the probability result and time taken (in seconds):")
    print(" - DPNL Basic Oracle    = Inference using the automatically-generated basic oracle")
    print(" - DPNL Complete Oracle = Inference using the complete hand-made oracle")
    print(" - DPNL Complete Oracle = Inference using the complete automatically-generated logic oracle based on"
          " Algorithm 3 of the paper")
    print(" - ProbLog              = Result from the ProbLog logic programming system")
    print("Format: <probability> (<time>s)")
    print()

    print(
        f"{'N':<3} | {'DPNL Basic Oracle':<25} | {'DPNL Hand-Crafted Oracle':<25} | {'DPNL Logic Oracle':<25} | {'ProbLog':<25}")
    print("-" * 100)

    for N in range(3, 10):
        graph = [[BoolRndVar("", random.uniform(0, 1)) for _ in range(N)] for _ in range(N)]

        dpnl_basic_result, dpnl_basic_time = run_with_timeout(run_dpnl, (graph, "basic"), timeout=90)
        dpnl_complete_result, dpnl_complete_time = run_with_timeout(run_dpnl, (graph, "complete"), timeout=90)
        problog_result, problog_time = run_problog(graph, timeout=90)
        dpnl_logic_result, dpnl_logic_time = run_with_timeout(run_dpnl, (graph, "logic"), timeout=90)

        print(f"{N:<3} | "
              f"{format_result(dpnl_basic_result, dpnl_basic_time)} | "
              f"{format_result(dpnl_complete_result, dpnl_complete_time)} | "
              f"{format_result(dpnl_logic_result, dpnl_logic_time)} | "
              f"{format_result(problog_result, problog_time)}")
