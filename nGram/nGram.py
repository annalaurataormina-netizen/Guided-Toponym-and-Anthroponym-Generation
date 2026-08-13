import json
import math
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

    def save(self):
        data = {
            f"{culture}|{context}": dict(counter)
            for (culture, context), counter in self.counts.items()
        }

        with open(f"nGram/models/{self.n}-gram_fit.json", "w") as f:
            json.dump(data, f)

    def load(self):
        with open(f"nGram/models/{self.n}-gram_fit.json", "r") as f:
            data = json.load(f)

        self.counts = defaultdict(Counter)

        for key, counter in data.items():
            culture, context = key.split("|", 1)
            self.counts[(culture, context)] = Counter(counter)

    '''
    x is a list or a tuple, where the first element is the name normalised, the second element is the culture
    example, x = ['Anna', 'Italian']
    '''

    def sequence_log_probability(self, x):
        name = '<' + x[0] + '>'
        culture = x[1]
        log_probability = 0.0
        count = 0

        for i in range(len(name) - self.n + 1):
            context = name[i:i + self.n - 1]
            next_char = name[i + self.n - 1]
            total = sum(self.counts[(culture, context)].values())
            if total == 0 or self.counts[(culture, context)][next_char] == 0:
                return float('-inf')
            count += 1
            log_probability += math.log(self.counts[(culture, context)][next_char] / total)

        return log_probability / count

    '''
    x is a string, after normalisation
    example, x = 'Anna'
    '''

    def sequence_log_probability_per_culture(self, name):
        cultures = set([culture for (culture, context) in self.counts.keys()])
        log_probabilities = {}

        for culture in cultures:
            log_probabilities[culture] = self.sequence_log_probability((name, culture))

        return log_probabilities
