from typing import List, Union

import torch
from numpy.random import choice
from problog.engine import DefaultEngine
from problog.extern import problog_export, problog_export_nondet
from problog.logic import (
    Term,
    AnnotatedDisjunction,
    term2list,
    Clause,
    And,
    Or,
    Var,
    Constant,
    list2term,
)
from problog.program import SimpleProgram

from deepproblog.engines.engine import Engine
from deepproblog.network import Network

# EXTERN = '{}_extern_'
EXTERN = "{}_extern_nocache_"


def wrap_tensor(x, store, name: Union[Term, List[Term]] = None):
    if type(x) is list:
        if name is None:
            return list2term([wrap_tensor(e, store) for e in x])
        else:
            return list2term([wrap_tensor(x[i], store, name[i]) for i in range(len(x))])
    else:
        if name is None:
            return Term("tensor", Constant(store.store(x)))
        else:
            return Term("tensor", store.store(x, name))


def create_with_substitution(formula, second, translation, key):
    if key in translation:
        return translation[key]
    node = second.get_node(key)
    t = type(node).__name__
    if t == "conj":
        return formula.add_and(
            [
                create_with_substitution(formula, second, translation, c)
                for c in node.children
            ],
            name=node.name,
        )
    elif t == "disj":
        return formula.add_or(
            [
                create_with_substitution(formula, second, translation, c)
                for c in node.children
            ],
            name=node.name,
        )
    else:
        raise (Exception("Unknown node ", node))


def get_predicate(net):
    def predicate(inputs):
        domain = net.domain
        output = net([term2list(inputs, False)])[0]
        net.cache[inputs] = output
        if net.eval_mode:
            _, result = torch.topk(output, net.k, 0)
            result = [domain[int(r)] for r in result]
        else:
            result = choice(
                domain, min(net.k, len(domain)), False, output.detach().numpy()
            ).tolist()
        return result

    return predicate


def get_det_predicate(net: Network, engine: Engine):
    def det_predicate(arguments):
        output = net([term2list(arguments, False)])[0]
        tensor_name = Term("nn", Term(net.name), arguments)
        # FIXME named tensor outputs for lists
        return wrap_tensor(output, engine.tensor_store, name=tensor_name)

    return det_predicate


class ExactEngine(Engine):
    def __init__(self, model):
        Engine.__init__(self, model)
        self.engine = DefaultEngine()

    def prepare(self, db):
        """ translates a program in Problog (line-by-line)
        Each element is a Problog Term or Clause
        """
        translated = SimpleProgram()
        for e in db:
            new_es = [e]
            if type(e) is Term or type(e) is Clause:
                p = e.probability  # if the term has a fixed probability
                if p is not None and p.functor == "nn":  # in case it's a neural predicate
                    if len(p.args) == 4:  # additive or probabilistic predicate
                        new_es = self.create_nn_predicate_ad(e)
                    elif len(p.args) == 3:
                        new_es = self.create_nn_predicate_det(e)
                    elif len(p.args) == 2:
                        new_es = self.create_nn_predicate_fact(e)
                    else:
                        raise ValueError(
                            "A neural predicate with {} arguments is not supported.".format(
                                len(p.args)
                            )
                        )
            for new_e in new_es:
                translated.add_clause(new_e)
        translated.add_clause(
            Clause(
                Term("_directive"),
                Term("use_module", Term("library", Term("lists.pl"))),
            )
        )
        clause_db = self.engine.prepare(translated)
        problog_export.database = clause_db
        for network in self.model.networks:
            if self.model.networks[network].det:
                signature = ["+term", "-term"]
                func = get_det_predicate(self.model.networks[network], self)
                problog_export(*signature)(
                    func, funcname=EXTERN.format(network), modname=None
                )
            elif self.model.networks[network].k is not None:
                signature = ["+term", "-list"]
                problog_export(*signature)(
                    get_predicate(self.model.networks[network]),
                    funcname="{}_extern_nocache_".format(network),
                    modname=None,
                )

        return clause_db

    def ground(self, query, label=None, repeat=1, **kwargs):
        db = self.model.solver.program
        if not self.model.solver.cache.cache:
            query = query.substitute()
        ground = self.engine.ground(db, query.query, label=label)
        return ground

    def create_nn_predicate_ad(self, e):
        """
        initializes a predicate described in the program with a neural network object
        add to the terms in the program a new probabilistic fact for each outcome of the network
        """
        p = e.probability
        net, inputs, output, domain = p.args
        network = self.model.networks[str(net)]

        # create in the program for each possible output in the domain a prolog term e.g. nn(mnist_net,[X],0)::digit(X,0),
        network.domain = term2list(domain, False)
        new_terms = []
        heads = []
        for j, domain_element in enumerate(term2list(domain, False)):
            head = Term(
                e.functor if network.k is None else e.functor + "_AD",
                *e.args,
                p=p.with_args(net, inputs, Constant(j))
            )
            head = head.apply_term({output: domain_element})
            heads.append(head)

        if type(e) is Clause:
            ad = AnnotatedDisjunction(heads, e.body)

        else:
            ad = Or.from_list(heads) # append with an or

        new_terms.append(ad)

        # if top-k configuration, enable to filter these terms
        if network.k is not None:
            body = []
            head = e.with_probability(None)
            if type(e) is Clause:
                body.append(e.body)
            body.append(Term(EXTERN.format(net), inputs, Var("Selected")))
            body.append(Term("member", output, Var("Selected")))
            body.append(Term(e.functor + "_AD", *e.args))
            new_e = Clause(head, And.from_list(body))
            new_terms.append(new_e)
        return new_terms

    def create_nn_predicate_fact(self, e):
        p = e.probability
        net, inputs = p.args
        network = self.model.networks[str(net)]
        return [e]

    def create_nn_predicate_det(self, e):
        p = e.probability
        net, inputs, output = p.args
        network = self.model.networks[str(net)]
        network.det = True
        if network.k is not None:
            raise ValueError(
                "k should be None for deterministc network {}".format(str(net))
            )
        head = e.with_probability(None)
        body = Term(EXTERN.format(net), inputs, output)
        return [Clause(head, body)]

    def register_foreign(self, func, function_name, arity_in, arity_out):
        signature = ["+term"] * arity_in + ["-term"] * arity_out
        problog_export.database = self.model.solver.program
        problog_export(*signature)(func, funcname=function_name, modname=None)

    def register_foreign_nondet(self, func, function_name, arity_in, arity_out):
        signature = ["+term"] * arity_in + ["-term"] * arity_out
        problog_export.database = self.model.solver.program
        problog_export_nondet(*signature)(func, funcname=function_name, modname=None)

    def get_hyperparameters(self) -> dict:
        return {"type": "ExactEngine"}

    @staticmethod
    def can_cache() -> bool:
        return True
