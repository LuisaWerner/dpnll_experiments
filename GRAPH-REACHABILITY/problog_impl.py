import time

import dpnl_impl


def new_program(I: dpnl_impl.GraphReachInput):
    size = I.size
    src = I.src
    dst = I.dst
    graph = I.graph

    # Generate edge facts
    facts = ""
    for i in range(size):
        for j in range(size):
            if i != j:
                if graph[i][j].value is True:
                    facts += f"edge({i},{j}).\n"
                elif not graph[i][j].defined():
                    facts += f"{graph[i][j].domain_distrib[True]}::edge({i},{j}).\n"

    # Reachability rules and query
    rules = f"""
        reachable(X,Y) :- edge(X,Y).
        reachable(X,Y) :- edge(X,Z), reachable(Z,Y).
        query(reachable({src},{dst})).
        """

    full_program = facts + rules

    return full_program


if __name__ == "__main__":
    import tools
    for length in range(2, 10):
        I = dpnl_impl.GraphReachInput(length, 0, 1)
        I.randomize_probabilities()
        t = time.time()
        result = tools.problog_interface.problog_run(new_program(I), 120)
        print(f"result : {result} ({time.time() - t} s.)")
