import json
import math
from collections import defaultdict, Counter


class nGram:

    def __init__(self, n: int):
        self.n = n
        self.counts = defaultdict(Counter)
        self.context_totals = {}
        self.vocab_size = 0

    def fit(self, x):

        for name, culture in x:
            name = '<' + name + '>'

            for i in range(len(name) - self.n + 1):
                context = name[i:i + self.n - 1]
                next_char = name[i + self.n - 1]

                self.counts[(culture, context)][next_char] += 1

        self._precompute_statistics()

    def _precompute_statistics(self):

        # Total number of observations for each (culture, context)
        self.context_totals = {
            key: sum(counter.values())
            for key, counter in self.counts.items()
        }

        # Vocabulary only needs to be calculated once
        vocabulary = set()

        for counter in self.counts.values():
            vocabulary.update(counter.keys())

        self.vocab_size = len(vocabulary)

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

        self._precompute_statistics()

    def sequence_log_probability(self, x):

        name = '<' + x[0] + '>'
        culture = x[1]

        alpha = 0.1

        log_probability = 0.0
        count = 0

        for i in range(len(name) - self.n + 1):

            context = name[i:i + self.n - 1]
            next_char = name[i + self.n - 1]

            key = (culture, context)

            counter = self.counts.get(key)

            if counter is None:
                total = 0
                next_char_count = 0
            else:
                total = self.context_totals[key]
                next_char_count = counter.get(next_char, 0)

            probability = (
                (next_char_count + alpha)
                / (total + alpha * self.vocab_size)
            )

            log_probability += math.log(probability)
            count += 1

        if count == 0:
            return None

        return log_probability / count

    def sequence_log_probability_per_culture(self, name):
        cultures = set([culture for (culture, context) in self.counts.keys()])
        log_probabilities = {}

        for culture in cultures:
            log_probabilities[culture] = self.sequence_log_probability((name, culture))

        return log_probabilities

    def next_char_log_probabilities(self, prefix, cultures, vocabulary, weights=None):
        alpha = 0.1
        context = prefix[-self.n + 1:]

        if weights is None:
            weights = [1.0 / len(cultures)] * len(cultures)

        log_probs = []

        for char in vocabulary:

            combined_log_prob = 0.0

            for culture, weight in zip(cultures, weights):

                key = (culture, context)
                counter = self.counts.get(key)

                if counter is None:
                    total = 0
                    char_count = 0
                else:
                    total = self.context_totals[key]
                    char_count = counter.get(char, 0)

                probability = (
                    (char_count + alpha)
                    / (total + alpha * self.vocab_size)
                )

                combined_log_prob += weight * math.log(probability)

            log_probs.append(combined_log_prob)

        return log_probs