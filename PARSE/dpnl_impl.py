from typing import Any

from dpnl.core.symbolic import Input, Symbolic
from dpnl.core.variable import RndVar, unknown
from dpnl.core.problem import PNLProblem
from dpnl.oracles.enumeration import EnumerationOracle
from dpnl.oracles.basic import BasicOracle
from dpnl.oracles.logic import LogicOracle
from dpnl.logics.sat_logic import SATLogic, SATLogicSymbolic
from dpnl.cachings.sat_logic_hash import SATCachingHash


class FiniteStateAutomaton:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        """
        Initialize the finite state automaton.

        :param states: A set of states.
        :param alphabet: A set of input symbols.
        :param transitions: A dictionary mapping (state, symbol) to a set of next states.
        :param start_state: The initial state.
        :param accept_states: A set of accepting (final) states.
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # {(state, symbol): set of next states}
        self.start_state = start_state
        self.accept_states = accept_states

    def process(self, word: list):
        """
        Process the input work and determine if it's accepted by the automaton.

        :param word: The list of symbol representing the word to process.
        :return: True if accepted, False otherwise.
        """
        current_states = {self.start_state}
        for symbol in word:
            next_states = set()
            for state in current_states:
                key = (state, symbol)
                if key in self.transitions:
                    next_states.update(self.transitions[key])
            current_states = next_states
            if not current_states:
                break
        return bool(current_states & self.accept_states)

    def process_rnd_word(self, rnd_word: list):
        """
        Process the input work and determine if it's accepted by the automaton.

        :param var_list: The list of random variable representing the random word to process.
        :return: True if accepted, False otherwise.
        """
        current_states = {self.start_state}
        for symbol_var in rnd_word:
            next_states = set()
            for state in current_states:
                key = (state, symbol_var.value)
                if key in self.transitions:
                    next_states.update(self.transitions[key])
            current_states = next_states
            if not current_states:
                break
        return bool(current_states & self.accept_states)


class ParseInput(Input):
    def __init__(self, A: FiniteStateAutomaton, length: int):
        p = 1.0 / len(A.alphabet)
        self.word = [RndVar(("word", idx), {symbol: p for symbol in A.alphabet}) for idx in range(length)]
        self.A = A
        super().__init__(probabilistic_attributes={"word"})


class ParseSymbolic:
    def __init__(self):
        super().__init__()

    def __call__(self, I: ParseInput):
        return I.A.process_rnd_word(I.word)


# Define the components of the DFA
states = {'q0', 'q1', 'q2'}
alphabet = {'a', 'b'}
transitions = {
    ('q0', 'a'): {'q1'},
    ('q0', 'b'): {'q0'},
    ('q1', 'a'): {'q1'},
    ('q1', 'b'): {'q2'},
    ('q2', 'a'): {'q1'},
    ('q2', 'b'): {'q0'},
}
start_state = 'q0'
accept_states = {'q2', 'q1'}

# Create the DFA instance
dfa = FiniteStateAutomaton(states, alphabet, transitions, start_state, accept_states)

# Test the DFA with some input strings
test_strings = ['ab', 'aab', 'baba', 'abab', 'baaa']
for s in test_strings:
    result = dfa.process(s)
    print(f"Input: {s} => Accepted: {result}")
