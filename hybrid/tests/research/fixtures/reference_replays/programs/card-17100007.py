import json, random, sys
sys.setrecursionlimit(20000)

def is_nt(s):
    return s[0] == 'n'

# ---------- CARD: plain lists, occurrences found by scanning, uses by scanning, repeat to stability ----------
class LNode:
    __slots__ = ('sym',)
    def __init__(self, sym):
        self.sym = sym

class CardGrammar:
    def __init__(self):
        self.rules = {0: []}
        self.next_id = 1
        self.global_fixes = 0
    def locate(self, node):
        for rid, body in self.rules.items():
            for i, n in enumerate(body):
                if n is node:
                    return rid, i
        return None
    def uses(self, X):
        return sum(1 for b in self.rules.values() for n in b if n.sym == ('n', X))
    def others(self, rid, i):
        body = self.rules[rid]
        d0, d1 = body[i].sym, body[i + 1].sym
        c = []
        for r2 in sorted(self.rules):
            b2 = self.rules[r2]
            for j in range(len(b2) - 1):
                if b2[j].sym == d0 and b2[j + 1].sym == d1:
                    if r2 == rid and abs(j - i) <= 1:
                        continue
                    c.append((r2, j))
        return c
    def check(self, node):
        loc = self.locate(node)
        if loc is None:
            return False
        rid, i = loc
        if i + 1 >= len(self.rules[rid]):
            return False
        c = self.others(rid, i)
        if not c:
            return False
        r2, j = min(c)
        self.match(self.rules[rid][i], self.rules[r2][j])
        return True
    def match(self, ss, m):
        rs, js = self.locate(ss)
        rm, jm = self.locate(m)
        d = (self.rules[rs][js].sym, self.rules[rs][js + 1].sym)
        if rm != 0 and len(self.rules[rm]) == 2:
            self.substitute(ss, rm)
        elif rs != 0 and len(self.rules[rs]) == 2:
            self.substitute(m, rs)
        else:
            R = self.next_id
            self.next_id += 1
            self.rules[R] = [LNode(d[0]), LNode(d[1])]
            self.substitute(m, R)
            self.substitute(ss, R)
        for s in d:
            if is_nt(s) and s[1] != 0 and s[1] in self.rules and self.uses(s[1]) == 1:
                self.expand(s[1])
    def substitute(self, node, R):
        rid, i = self.locate(node)
        body = self.rules[rid]
        q = body[i - 1] if i > 0 else None
        new = LNode(('n', R))
        body[i:i + 2] = [new]
        if q is not None and self.check(q):
            return
        self.check(new)
    def expand(self, X):
        loc = None
        for rid, body in self.rules.items():
            if rid == X:
                continue
            for i, n in enumerate(body):
                if n.sym == ('n', X):
                    loc = (rid, i)
        rid, i = loc
        body = self.rules[rid]
        xb = self.rules.pop(X)
        body[i:i + 1] = xb
        left = body[i - 1] if i > 0 else None
        last = xb[-1]
        if left is not None:
            self.check(left)
        self.check(last)
    def stabilize(self):
        for _ in range(10000):
            changed = False
            for rid in sorted(self.rules):
                body = self.rules[rid]
                for i in range(len(body) - 1):
                    if self.others(rid, i):
                        self.check(body[i])
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                for rid in sorted(self.rules):
                    if rid == 0:
                        continue
                    u = self.uses(rid)
                    if u <= 1:
                        if u == 1:
                            self.expand(rid)
                        else:
                            del self.rules[rid]
                        changed = True
                        break
            if not changed:
                return
            self.global_fixes += 1
    def append(self, x):
        S = self.rules[0]
        S.append(LNode(('t', x)))
        if len(S) >= 2:
            self.check(S[-2])
        self.stabilize()
    def grammar(self):
        return {rid: [n.sym for n in b] for rid, b in self.rules.items()}

# ---------- COMPARATOR: SEQUITUR with doubly linked rules, digram hash index, reference counts ----------
class DNode:
    __slots__ = ('sym', 'prev', 'next', 'owner', 'alive')
    def __init__(self, sym, owner):
        self.sym = sym
        self.prev = None
        self.next = None
        self.owner = owner
        self.alive = True

class Rule:
    __slots__ = ('first', 'last')
    def __init__(self):
        self.first = None
        self.last = None

class Sequitur:
    def __init__(self):
        self.rules = {0: Rule()}
        self.index = {}
        self.count = {}
        self.refs = {}
        self.next_id = 1
    def key(self, n):
        return (n.sym, n.next.sym)
    def add_dig(self, n):
        if n is not None and n.next is not None:
            self.index.setdefault(self.key(n), set()).add(n)
    def rem_dig(self, n):
        if n is not None and n.next is not None:
            s = self.index.get(self.key(n))
            if s is not None:
                s.discard(n)
    def pos(self, n):
        k = 0
        while n.prev is not None:
            n = n.prev
            k += 1
        return k
    def ref_add(self, n):
        if is_nt(n.sym):
            X = n.sym[1]
            self.count[X] = self.count.get(X, 0) + 1
            self.refs.setdefault(X, set()).add(n)
    def ref_del(self, n):
        if is_nt(n.sym):
            X = n.sym[1]
            self.count[X] -= 1
            self.refs[X].discard(n)
    def check(self, n):
        if not n.alive or n.next is None:
            return False
        c = [m for m in self.index.get(self.key(n), ()) if m is not n and m.next is not n and n.next is not m]
        if not c:
            return False
        m = min(c, key=lambda z: (z.owner, self.pos(z)))
        self.match(n, m)
        return True
    def is_body(self, n):
        return n.owner != 0 and n.prev is None and n.next is not None and n.next.next is None
    def match(self, ss, m):
        d = (ss.sym, ss.next.sym)
        if self.is_body(m):
            self.substitute(ss, m.owner)
        elif self.is_body(ss):
            self.substitute(m, ss.owner)
        else:
            R = self.next_id
            self.next_id += 1
            r = Rule()
            a = DNode(d[0], R)
            b = DNode(d[1], R)
            a.next = b
            b.prev = a
            r.first = a
            r.last = b
            self.rules[R] = r
            self.count[R] = 0
            self.refs[R] = set()
            self.ref_add(a)
            self.ref_add(b)
            self.add_dig(a)
            self.substitute(m, R)
            self.substitute(ss, R)
        for s in d:
            if is_nt(s) and s[1] != 0 and s[1] in self.rules and self.count[s[1]] == 1:
                self.expand(s[1])
    def substitute(self, n, R):
        n2 = n.next
        q = n.prev
        nxt = n2.next
        owner = n.owner
        self.rem_dig(q)
        self.rem_dig(n)
        self.rem_dig(n2)
        for z in (n, n2):
            z.alive = False
            self.ref_del(z)
        new = DNode(('n', R), owner)
        new.prev = q
        new.next = nxt
        r = self.rules[owner]
        if q is None:
            r.first = new
        else:
            q.next = new
        if nxt is None:
            r.last = new
        else:
            nxt.prev = new
        self.ref_add(new)
        self.add_dig(q)
        self.add_dig(new)
        if q is not None and self.check(q):
            return
        self.check(new)
    def expand(self, X):
        p = next(iter(self.refs[X]))
        r = self.rules[X]
        f = r.first
        l = r.last
        left = p.prev
        right = p.next
        owner = p.owner
        self.rem_dig(left)
        self.rem_dig(p)
        p.alive = False
        self.ref_del(p)
        del self.rules[X]
        del self.count[X]
        del self.refs[X]
        z = f
        while z is not None:
            z.owner = owner
            z = z.next
        ro = self.rules[owner]
        f.prev = left
        l.next = right
        if left is None:
            ro.first = f
        else:
            left.next = f
        if right is None:
            ro.last = l
        else:
            right.prev = l
        self.add_dig(left)
        self.add_dig(l)
        if left is not None:
            self.check(left)
        self.check(l)
    def append(self, x):
        r = self.rules[0]
        n = DNode(('t', x), 0)
        if r.last is None:
            r.first = n
            r.last = n
        else:
            n.prev = r.last
            r.last.next = n
            r.last = n
            self.add_dig(n.prev)
            self.check(n.prev)
    def grammar(self):
        out = {}
        for rid, r in self.rules.items():
            s = []
            z = r.first
            while z is not None:
                s.append(z.sym)
                z = z.next
            out[rid] = s
        return out

def canon(g):
    names = {0: 'S'}
    order = [0]
    k = 0
    while k < len(order):
        for s in g[order[k]]:
            if is_nt(s) and s[1] not in names:
                names[s[1]] = 'R%d' % len(names)
                order.append(s[1])
        k += 1
    def rend(s):
        return names[s[1]] if is_nt(s) else s[1]
    return (tuple((names[r], tuple(rend(s) for s in g[r])) for r in order), len(g))

def expand_str(g, r=0):
    return ''.join(expand_str(g, s[1]) if is_nt(s) else s[1] for s in g[r])

rng = random.Random(0)
seqs = ['abab', 'abcabc', 'aaaa', 'aaaaaaaaab', 'abababab', 'abcabdabcabd', 'abbbabcbb']
for _ in range(60):
    L = rng.randint(12, 40)
    A = rng.choice(['ab', 'abc', 'abcd'])
    seqs.append(''.join(rng.choice(A) for _ in range(L)))

max_diff = 0.0
exp_diff = 0.0
errors = 0
global_fixes = 0
final = {}
for seq in seqs:
    c = CardGrammar()
    q = Sequitur()
    for t, x in enumerate(seq):
        try:
            c.append(x)
            q.append(x)
            gc = c.grammar()
            gq = q.grammar()
            d = 0.0 if canon(gc) == canon(gq) else 1.0
            e = 0.0 if (expand_str(gc) == seq[:t + 1] and expand_str(gq) == seq[:t + 1]) else 1.0
        except Exception:
            errors += 1
            d = 1.0
            e = 1.0
        max_diff = max(max_diff, d)
        exp_diff = max(exp_diff, e)
        if d == 1.0 and e == 1.0 and errors:
            break
    global_fixes += c.global_fixes
    try:
        final[seq] = canon(q.grammar())
    except Exception:
        final[seq] = None

w1 = canon({0: [('n', 1), ('n', 1)], 1: [('t', 'a'), ('t', 'b')]})
w2 = canon({0: [('n', 1), ('n', 1)], 1: [('t', 'a'), ('t', 'b'), ('t', 'c')]})
wit = 0.0 if (final.get('abab') == w1 and final.get('abcabc') == w2) else 1.0

report = {
    'comparisons': [
        {'declared_id': 'sequitur_published',
         'comparator': 'SEQUITUR with doubly linked rule bodies, digram hash index (all occurrences), rule reference counts, check/match/substitute/expand scheduling of section 4',
         'family': 'other_known',
         'transformations': [],
         'limit_only': False,
         'reference': 'Nevill-Manning and Witten (1997), Identifying Hierarchical Structure in Sequences: A linear-time algorithm, JAIR 7:67-82, doi:10.1613/jair.374',
         'inputs_tested': len(seqs),
         'max_abs_diff': max_diff},
        {'declared_id': None,
         'comparator': 'sanity: expansion of S equals input prefix for card and comparator',
         'family': 'other_known',
         'transformations': [],
         'limit_only': False,
         'reference': 'card readout requirement',
         'inputs_tested': len(seqs),
         'max_abs_diff': exp_diff},
        {'declared_id': None,
         'comparator': 'sanity: card witnesses abab -> S->AA,A->ab and abcabc -> S->BB,B->abc',
         'family': 'other_known',
         'transformations': [],
         'limit_only': False,
         'reference': 'card capability prediction',
         'inputs_tested': 2,
         'max_abs_diff': wit}
    ],
    'notes': 'Readout is a grammar, so the per-step difference is a 0/1 indicator: 0 if the card grammar and the comparator grammar are identical up to nonterminal renaming (canonical BFS naming from S, same rule count), 1 otherwise; already order one, no rescaling. Card is implemented with plain lists, scan-based occurrence search, scan-based use counts, and a global repeat-to-stability pass after each token; extra global fixes needed=%d; runtime errors=%d.' % (global_fixes, errors)
}
print(json.dumps(report))
