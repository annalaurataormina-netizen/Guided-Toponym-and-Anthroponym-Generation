from collections import defaultdict
import random

from torch.utils.data import Sampler


class LabelBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, samples_per_class=4):
        """
        labels: list of language labels for every sample in the training set.
        batch_size: total batch size.
        samples_per_class: number of samples from each language in every batch.
        """

        assert batch_size % samples_per_class == 0

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class

        # Build mapping: language -> dataset indices
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        # Keep only languages with enough samples
        self.label_to_indices = {
            label: indices
            for label, indices in label_to_indices.items()
            if len(indices) >= samples_per_class
        }

        self.classes = list(self.label_to_indices.keys())

        self.num_batches = len(labels) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):

            batch = []

            # Randomly choose languages
            chosen_classes = random.sample(
                self.classes,
                min(self.classes_per_batch, len(self.classes))
            )

            # Pick samples from each language
            for cls in chosen_classes:
                indices = random.sample(
                    self.label_to_indices[cls],
                    self.samples_per_class
                )
                batch.extend(indices)

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches