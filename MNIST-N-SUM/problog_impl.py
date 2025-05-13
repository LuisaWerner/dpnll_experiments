import dpnl_impl


def new_program(I: dpnl_impl.MNISTInput):
    length = I.length
    result = I.result
    facts = ""

    num = [[], []]
    for i in range(2):
        for idx in range(length):
            pred = f"num_{i}_{idx}"
            annotated_dij = [f"{I.num[i][I.length-idx-1].domain_distrib[value]}::{pred}({value})" for value in range(10)]
            annotated_dij = "; ".join(annotated_dij)

            num[i].append(pred)
            facts += f"{annotated_dij}.\n"

    c = [f"carry_{idx}" for idx in range(length + 1)]
    d = [f"d_{idx}" for idx in range(length)]
    r = [f"result_{idx}" for idx in range(length + 1)]

    rules = f"{c[0]}(0).\n"
    for idx in range(length):
        rules += f"{d[idx]}(Z) :- {num[0][idx]}(X), {num[1][idx]}(Y), {c[idx]}(C), Z is X+Y+C.\n"
        rules += f"{r[idx]}(Z) :- {d[idx]}(Z), Z < 10.\n"
        rules += f"{r[idx]}(Z) :- {d[idx]}(X), X > 9, Z is X-10.\n"
        rules += f"{c[idx + 1]}(1) :- {d[idx]}(X), X > 9.\n"
        rules += f"{c[idx + 1]}(0) :- {d[idx]}(X), X < 10.\n"

    rules += f"{r[length]}(Z) :- {c[length]}(Z).\n"

    result_digits = []
    for idx in range(length + 1):
        result_digits.append(result % 10)
        result = int(result / 10)
    success = [f"{r[idx]}({result_digits[idx]})" for idx in range(length + 1)]
    rules += f"success :- {', '.join(success)}.\n"

    return facts + rules + "query(success).\n"


if __name__ == "__main__":
    import tools

    for i in range(10):
        print(tools.problog_interface.problog_run(new_program(6, 100), i * 2))
