import random





def unit_propagate(clauses: list[list[int]]):
    """
    Performs unit propagation on a CNF formula.

    Args:
        clauses (List[List[int]]): A list of clauses, each clause is a list of integers representing literals.

    Returns:
        Tuple[Optional[List[List[int]]], dict]: A tuple containing the simplified list of clauses and the current assignments.
            If a conflict is detected (i.e., an empty clause is produced), the clauses list is None.
    """
    assignments = {}
    while True:
        unit_clauses = [c for c in clauses if len(c) == 1]
        if not unit_clauses:
            break  # No more unit clauses to process

        for unit in unit_clauses:
            literal = unit[0]
            var = abs(literal)
            value = literal > 0

            # Check for conflicting assignments
            if var in assignments:
                if assignments[var] != value:
                    # Conflict detected
                    return [[]], assignments
            else:
                assignments[var] = value

            new_clauses = []
            for clause in clauses:
                if literal in clause:
                    continue  # Clause is satisfied
                if -literal in clause:
                    new_clause = [l for l in clause if l != -literal]
                    if not new_clause:
                        # Conflict detected
                        return [[]], assignments
                    new_clauses.append(new_clause)
                else:
                    new_clauses.append(clause)
            clauses = new_clauses

    return clauses, assignments


"""for _ in range(1000):
    N = random.randint(10, 20)
    clauses = []
    for _ in range(N):
        M = random.randint(1, 5)
        clauses.append([random.choice(list(range(-10, 0)) + list(range(11))) for _ in range(M)])
    r1 = unit_propagation(clauses)
    r2, assign = unit_propagate(clauses)
    if not (to_string(r1) == to_string(r2)):
        print(clauses)
        print(r1)
        print(r2)
        print(assign)
        print("-"*100)"""