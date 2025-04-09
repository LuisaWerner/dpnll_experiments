import multiprocessing
import random
import time
import tempfile
import subprocess
from dpnl import (
    BoolRndVar, PNLProblem,
    basic_oracle, basic_oracle_choose_heuristic, unknown
)
from graph_reachability import (
    graph_reachability, graph_reachability_complete_oracle,
    graph_reachability_complete_oracle_choose_heuristic, random_graph
)
import prolog_logic


# === DPNL Run Function ===

def run_dpnl(graph, mode):
    N = len(graph)
    pnl_problem = PNLProblem((graph, 0, 1), graph_reachability)

    if mode == "basic":
        oracle = basic_oracle(graph_reachability)
        choose = basic_oracle_choose_heuristic
    elif mode == "complete":
        oracle = graph_reachability_complete_oracle
        choose = graph_reachability_complete_oracle_choose_heuristic
    elif mode == "logic":
        X = ()
        for i in range(N):
            for j in range(N):
                X += (graph[i][j],)
        pnl_problem = PNLProblem(X, prolog_logic.graph_reachability_S(N, 0, 1))
        oracle = pnl_problem.S.logic.Oracle(pnl_problem.S)
        choose = prolog_logic.choose_heuristic
    else:
        raise ValueError("Unknown mode")

    prob = pnl_problem.prob(oracle, True, choose)
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
            elapsed_time = time.time() - start_time
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

    print("\n=== 🔍 Comparison: DPNL vs. ProbLog on Graph Reachability ===")
    print("Each row corresponds to a random NxN graph (N = number of nodes).")
    print("Each each edge have a certain probability of being inside the graph.")
    print("We estimate the probability that there is a path from node 0 to node 1.")
    print("Columns show the probability estimate and time taken (in seconds):")
    print(" - DPNL Basic Oracle    = Inference using the simple oracle")
    print(" - DPNL Complete Oracle = Inference using the complete hand-made oracle")
    print(" - DPNL Complete Oracle = Inference using the complete logic oracle using Algorithm 3 of the paper")
    print(" - ProbLog              = Result from the ProbLog logic programming system")
    print("Format: <probability> (<time>s)")
    print()

    print(
        f"{'N':<3} | {'DPNL Basic Oracle':<25} | {'DPNL Hand-Crafted Oracle':<25} | {'DPNL Logic Oracle':<25} | {'ProbLog':<25}")
    print("-" * 100)

    for N in range(3, 10):
        graph = [[BoolRndVar("", random.uniform(0, 1)) for _ in range(N)] for _ in range(N)]

        # DPNL (basic)
        dpnl_basic_result, dpnl_basic_time = run_with_timeout(run_dpnl, (graph, "basic"), timeout=30)

        # DPNL (hand-crafted)
        dpnl_complete_result, dpnl_complete_time = run_with_timeout(run_dpnl, (graph, "complete"), timeout=30)

        # ProbLog (with internal timeout handling)
        problog_result, problog_time = run_problog(graph, timeout=30)

        # DPNL (logic based)
        dpnl_logic_result, dpnl_logic_time = run_with_timeout(run_dpnl, (graph, "logic"), timeout=30)

        print(f"{N:<3} | "
              f"{format_result(dpnl_basic_result, dpnl_basic_time)} | "
              f"{format_result(dpnl_complete_result, dpnl_complete_time)} | "
              f"{format_result(dpnl_logic_result, dpnl_logic_time)} | "
              f"{format_result(problog_result, problog_time)}")
