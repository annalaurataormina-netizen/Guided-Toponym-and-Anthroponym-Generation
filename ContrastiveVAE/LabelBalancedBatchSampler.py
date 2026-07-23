from collections import defaultdict
import math
import random

import numpy as np
import torch
from torch.utils.data import Sampler


class LabelBalancedBatchSampler(Sampler):
    def __init__(self, labels: list[int], batch_size: int, samples_per_class: int=4):
        """
        Parameters
        ----------
        labels : list[int]
            Class label for every sample in the dataset.

        batch_size : int
            Total batch size.

        samples_per_class : int
            Number of samples drawn from each class in every batch.
        """

        assert batch_size % samples_per_class == 0

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class

        # Build mapping: class -> dataset indices
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        self.label_to_indices = {
            label: indices
            for label, indices in label_to_indices.items()
            if len(indices) >= samples_per_class
        }

        self.label_to_indices = dict(label_to_indices)
        self.classes = list(self.label_to_indices.keys())

        # sqrt(class_size) sampling probabilities
        weights = np.array(
            [math.sqrt(len(self.label_to_indices[c])) for c in self.classes],
            dtype=np.float64,
        )
        self.class_probabilities = weights / weights.sum()

        # Keep roughly the same epoch length as standard training
        self.num_batches = len(labels) // batch_size

    def __iter__(self):

        # One shuffled queue per class
        pools = {}
        pointers = {}

        for cls, indices in self.label_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            pools[cls] = shuffled
            pointers[cls] = 0

        for _ in range(self.num_batches):

            batch = []

            # Sample classes without replacement
            chosen_classes = np.random.choice(
                self.classes,
                size=min(self.classes_per_batch, len(self.classes)),
                replace=False,
                p=self.class_probabilities,
            )

            for cls in chosen_classes:

                # Reshuffle this class only after it has been exhausted
                if pointers[cls] + self.samples_per_class > len(pools[cls]):
                    random.shuffle(pools[cls])
                    pointers[cls] = 0

                start = pointers[cls]
                end = start + self.samples_per_class

                batch.extend(pools[cls][start:end])

                pointers[cls] = end

            random.shuffle(batch)

            print(
                "classes selected:",
                len(chosen_classes),
                "samples selected:",
                len(batch)
            )

            assert len(batch) == self.batch_size, (
                f"Wrong batch size: got {len(batch)}, expected {self.batch_size}"
            )

            yield batch

    def __len__(self) -> int:
        return self.num_batches