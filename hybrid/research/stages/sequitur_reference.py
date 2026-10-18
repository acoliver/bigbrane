"""SEQUITUR digram uniqueness and rule utility (Nevill-Manning and Witten, JAIR 1997, section 4).

A scan-based host reference prioritizes specification and deterministic occurrence order over linear runtime.
Rule names are serialized in breadth-first order, so incidental allocation ids cannot decide equality.
"""
from itertools import count

import numpy as np


def grammar_vector(grammar, width):
    names = {0: 0}
    order = [0]
    for rule in order:
        for kind, symbol in grammar[rule]:
            if kind == 'n' and symbol not in names:
                names[symbol] = len(names)
                order.append(symbol)
    result = [len(grammar)]
    for rule in order:
        result.append(len(grammar[rule]))
        result.extend(-names[symbol] - 1 if kind == 'n' else ord(symbol) + 1
                      for kind, symbol in grammar[rule])
    if len(result) > width:
        raise ValueError('grammar serialization exceeds the input-derived bound')
    return result + [0] * (width - len(result))


def sequitur_outputs(inputs):
    ids = count()
    rule_ids = count(1)
    rules = {0: []}

    def node(symbol):
        return next(ids), symbol

    def locate(identity):
        return next(((r, i) for r, body in rules.items() for i, n in enumerate(body) if n[0] == identity), None)

    def uses(rule):
        return [(r, i) for r, body in rules.items() for i, n in enumerate(body) if n[1] == ('n', rule)]

    def check(identity):
        position = locate(identity)
        if position is None:
            return False
        r, i = position
        if i + 1 == len(rules[r]):
            return False
        digram = [n[1] for n in rules[r][i:i + 2]]
        matches = [(s, j) for s, body in sorted(rules.items()) for j in range(len(body) - 1)
                   if (s != r or abs(i - j) > 1) and [n[1] for n in body[j:j + 2]] == digram]
        if not matches:
            return False
        s, j = matches[0]
        other = rules[s][j][0]
        if s != 0 and len(rules[s]) == 2:
            substitute(identity, s)
        elif r != 0 and len(rules[r]) == 2:
            substitute(other, r)
        else:
            new_rule = next(rule_ids)
            rules[new_rule] = [node(symbol) for symbol in digram]
            substitute(other, new_rule)
            substitute(identity, new_rule)
        for kind, rule in digram:
            if kind == 'n' and rule in rules and len(uses(rule)) == 1:
                expand(rule)
        return True

    def substitute(identity, rule):
        r, i = locate(identity)
        previous = rules[r][i - 1][0] if i else None
        replacement = node(('n', rule))
        rules[r][i:i + 2] = [replacement]
        if previous is not None and check(previous):
            return
        check(replacement[0])

    def expand(rule):
        (r, i), = uses(rule)
        body = rules.pop(rule)
        previous = rules[r][i - 1][0] if i else None
        rules[r][i:i + 1] = body
        if previous is not None:
            check(previous)
        check(body[-1][0])

    outputs = []
    for symbol in inputs:
        if not isinstance(symbol, str) or len(symbol) != 1:
            raise ValueError('SEQUITUR tokens must be single characters')
        rules[0].append(node(('t', symbol)))
        if len(rules[0]) >= 2:
            check(rules[0][-2][0])
        grammar = {r: [n[1] for n in body] for r, body in rules.items()}
        outputs.append(grammar_vector(grammar, 4 * len(inputs) + 4))
    return np.asarray(outputs, dtype=float)
