from collections import defaultdict, Counter


class nGram:

    def __init__(self, n: int):
        self.n = n
        self.counts = defaultdict(Counter)

    def fit(self, x):
        for name, culture in x:
            name = '<' + name + '>'
            for i in range(len(name) - self.n + 1):
                context = name[i:i + self.n - 1]
                next_char = name[i + self.n - 1]
                self.counts[(culture, context)][next_char] += 1

    def sequence_probability(self, x):
        name = '<' + x[0] + '>'
        culture = x[1]
        probability = 1.0
        count = 0

        for i in range(len(name) - self.n + 1):
            context = name[i:i + self.n - 1]
            next_char = name[i + self.n - 1]
            total = sum(self.counts[(culture, context)].values())
            if total == 0:
                return 0.0
            probability *= (self.counts[(culture, context)][next_char] / total)
            count += 1

        return probability ** (1 / count)
