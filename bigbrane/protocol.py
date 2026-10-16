"""Fixed version-one search grid and seed contract."""
from dataclasses import asdict, dataclass

FAMILIES = ('hmm', 'rnn', 'attention')
LEARNING_RATES = (.001, .01)


@dataclass(frozen=True)
class FitSpec:
    root: int
    stage: int
    regime: int
    generator: int
    size_index: int
    family: str
    replica: int
    candidate: int

    def __post_init__(self):
        if self.family not in FAMILIES or self.stage not in (0, 1):
            raise ValueError('invalid family or stage')
        if self.size_index not in range(4) or self.replica not in (0, 1) or self.candidate not in (0, 1):
            raise ValueError('invalid search coordinate')

    @property
    def key(self):
        return f's{self.stage}-r{self.regime}-g{self.generator}-n{self.size_index}-{self.family}-rep{self.replica}-c{self.candidate}'

    def seed(self, role, epoch=None):
        candidate = self.candidate if self.family == 'hmm' else 0
        values = (self.root, 3, self.stage, self.regime, self.generator, self.size_index,
                  FAMILIES.index(self.family), self.replica, candidate, role)
        return values if epoch is None else values+(epoch,)

    def record(self):
        return {**asdict(self), 'key': self.key, 'initialization_seed': self.seed(0),
                'shuffle_seeds': [self.seed(1, e) for e in range(1, 81)] if self.family != 'hmm' else [],
                'learning_rate': None if self.family == 'hmm' else LEARNING_RATES[self.candidate]}


def grid(root, stage, pilot=False):
    return [FitSpec(root, stage, regime, g, size, family, replica, candidate)
            for regime in (range(1) if pilot else range(2))
            for g in (range(1) if pilot else range(4))
            for size in ([0] if pilot else ([0, 1, 2, 3] if stage == 0 else [1, 3]))
            for family in FAMILIES
            for replica in (range(1) if pilot else range(2))
            for candidate in (range(1) if pilot else range(2))]


def selection_key(record):
    return (record['validation_nll'], record['checkpoint'], record['candidate'])
