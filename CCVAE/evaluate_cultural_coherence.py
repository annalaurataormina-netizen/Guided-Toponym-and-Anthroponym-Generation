import torch
from torch.utils.data import DataLoader

from ContrastiveVAE.NameDataset import NameDataset
from CultureClassifier.CultureClassifier import CultureClassifier


def evaluate_cultural_coherence(
        generated_per_language,
        language_to_id,
        device,
        vocab,
        names_normalised,
        culture_counts
):
    min_samples = 1000

    id_to_language = {id: language for language, id in language_to_id.items()}

    # Keep only cultures with at least min_samples training examples
    names_normalised = [
        [name, label]
        for name, label in names_normalised
        if culture_counts[id_to_language[label]] >= min_samples
    ]

    remaining_cultures = sorted(
        set(label for _, label in names_normalised)
    )

    old_to_new = {
        old: new
        for new, old in enumerate(remaining_cultures)
    }

    language_to_id = {
        language: old_to_new[label]
        for language, label in language_to_id.items()
        if label in old_to_new
    }

    num_cultures_filtered = len(language_to_id)

    batch_size, embed_dim, hidden_dim, num_layers, lr, epochs = (
        512, 32, 256, 1, 0.0005, 30
    )

    classifier = CultureClassifier(
        vocab,
        embed_dim,
        hidden_dim,
        num_layers,
        num_cultures_filtered
    )

    classifier_name = (
        f'CultureClassifier/models/'
        f'best_model_bs{batch_size}_ed{embed_dim}_hd{hidden_dim}_'
        f'nl{num_layers}_lr{lr}_ep{epochs}_ms{min_samples}.pt'
    )

    classifier.load_state_dict(
        torch.load(classifier_name, map_location=device)
    )

    classifier.to(device)
    classifier.eval()

    # Store (language, accuracy) so weighted averages are easy to calculate
    accuracies = []
    top2_accuracies = []
    top3_accuracies = []
    top4_accuracies = []
    top5_accuracies = []
    top6_accuracies = []
    top7_accuracies = []
    top8_accuracies = []
    top9_accuracies = []
    top10_accuracies = []

    with torch.no_grad():

        for language, label in sorted(
                language_to_id.items(),
                key=lambda x: x[1]
        ):

            generated_names = generated_per_language[language]

            data = [
                [name, label]
                for name in generated_names
            ]

            dataset = NameDataset(data, vocab)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )

            correct = 0
            top2_correct = 0
            top3_correct = 0
            top4_correct = 0
            top5_correct = 0
            top6_correct = 0
            top7_correct = 0
            top8_correct = 0
            top9_correct = 0
            top10_correct = 0
            total = 0

            for sequences, lengths, _ in dataloader:

                sequences = sequences.to(device)
                lengths = lengths.cpu()

                logits = classifier(sequences, lengths)

                predictions = logits.argmax(dim=1)

                top2_predictions = torch.topk(
                    logits, k=2, dim=1
                ).indices

                top3_predictions = torch.topk(
                    logits, k=3, dim=1
                ).indices

                top4_predictions = torch.topk(
                    logits, k=4, dim=1
                ).indices

                top5_predictions = torch.topk(
                    logits, k=5, dim=1
                ).indices

                top6_predictions = torch.topk(
                    logits, k=6, dim=1
                ).indices

                top7_predictions = torch.topk(
                    logits, k=7, dim=1
                ).indices

                top8_predictions = torch.topk(
                    logits, k=8, dim=1
                ).indices

                top9_predictions = torch.topk(
                    logits, k=9, dim=1
                ).indices

                top10_predictions = torch.topk(
                    logits, k=10, dim=1
                ).indices

                correct += (
                    (predictions == label)
                    .sum()
                    .item()
                )

                top2_correct += (
                    (top2_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top3_correct += (
                    (top3_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top4_correct += (
                    (top4_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top5_correct += (
                    (top5_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top6_correct += (
                    (top6_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top7_correct += (
                    (top7_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top8_correct += (
                    (top8_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top9_correct += (
                    (top9_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                top10_correct += (
                    (top10_predictions == label)
                    .any(dim=1)
                    .sum()
                    .item()
                )

                total += len(sequences)

            generation_accuracy = correct / total
            top2_generation_accuracy = top2_correct / total
            top3_generation_accuracy = top3_correct / total
            top4_generation_accuracy = top4_correct / total
            top5_generation_accuracy = top5_correct / total
            top6_generation_accuracy = top6_correct / total
            top7_generation_accuracy = top7_correct / total
            top8_generation_accuracy = top8_correct / total
            top9_generation_accuracy = top9_correct / total
            top10_generation_accuracy = top10_correct / total

            accuracies.append(
                (language, generation_accuracy)
            )
            top2_accuracies.append(
                (language, top2_generation_accuracy)
            )
            top3_accuracies.append(
                (language, top3_generation_accuracy)
            )
            top4_accuracies.append(
                (language, top4_generation_accuracy)
            )
            top5_accuracies.append(
                (language, top5_generation_accuracy)
            )
            top6_accuracies.append(
                (language, top6_generation_accuracy)
            )
            top7_accuracies.append(
                (language, top7_generation_accuracy)
            )
            top8_accuracies.append(
                (language, top8_generation_accuracy)
            )
            top9_accuracies.append(
                (language, top9_generation_accuracy)
            )
            top10_accuracies.append(
                (language, top10_generation_accuracy)
            )

    avg_generation_accuracy = sum(
        accuracy
        for _, accuracy in accuracies
    ) / len(accuracies)

    avg_top2_generation_accuracy = sum(
        accuracy
        for _, accuracy in top2_accuracies
    ) / len(top2_accuracies)

    avg_top3_generation_accuracy = sum(
        accuracy
        for _, accuracy in top3_accuracies
    ) / len(top3_accuracies)

    avg_top4_generation_accuracy = sum(
        accuracy
        for _, accuracy in top4_accuracies
    ) / len(top4_accuracies)

    avg_top5_generation_accuracy = sum(
        accuracy
        for _, accuracy in top5_accuracies
    ) / len(top5_accuracies)

    avg_top6_generation_accuracy = sum(
        accuracy
        for _, accuracy in top6_accuracies
    ) / len(top6_accuracies)

    avg_top7_generation_accuracy = sum(
        accuracy
        for _, accuracy in top7_accuracies
    ) / len(top7_accuracies)

    avg_top8_generation_accuracy = sum(
        accuracy
        for _, accuracy in top8_accuracies
    ) / len(top8_accuracies)

    avg_top9_generation_accuracy = sum(
        accuracy
        for _, accuracy in top9_accuracies
    ) / len(top9_accuracies)

    avg_top10_generation_accuracy = sum(
        accuracy
        for _, accuracy in top10_accuracies
    ) / len(top10_accuracies)

    total_weight = sum(
        culture_counts[language]
        for language, _ in accuracies
    )

    weighted_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in accuracies
    ) / total_weight

    weighted_top2_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top2_accuracies
    ) / total_weight

    weighted_top3_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top3_accuracies
    ) / total_weight

    weighted_top4_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top4_accuracies
    ) / total_weight

    weighted_top5_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top5_accuracies
    ) / total_weight

    weighted_top6_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top6_accuracies
    ) / total_weight

    weighted_top7_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top7_accuracies
    ) / total_weight

    weighted_top8_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top8_accuracies
    ) / total_weight

    weighted_top9_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top9_accuracies
    ) / total_weight

    weighted_top10_generation_accuracy = sum(
        accuracy * culture_counts[language]
        for language, accuracy in top10_accuracies
    ) / total_weight

    return {
        # Macro averages
        "Average generation accuracy":
            avg_generation_accuracy,

        "Average top-2 generation accuracy":
            avg_top2_generation_accuracy,

        "Average top-3 generation accuracy":
            avg_top3_generation_accuracy,

        "Average top-4 generation accuracy":
            avg_top4_generation_accuracy,

        "Average top-5 generation accuracy":
            avg_top5_generation_accuracy,

        "Average top-6 generation accuracy":
            avg_top6_generation_accuracy,

        "Average top-7 generation accuracy":
            avg_top7_generation_accuracy,

        "Average top-8 generation accuracy":
            avg_top8_generation_accuracy,

        "Average top-9 generation accuracy":
            avg_top9_generation_accuracy,

        "Average top-10 generation accuracy":
            avg_top10_generation_accuracy,

        # Weighted averages
        "Weighted generation accuracy":
            weighted_generation_accuracy,

        "Weighted top-2 generation accuracy":
            weighted_top2_generation_accuracy,

        "Weighted top-3 generation accuracy":
            weighted_top3_generation_accuracy,

        "Weighted top-4 generation accuracy":
            weighted_top4_generation_accuracy,

        "Weighted top-5 generation accuracy":
            weighted_top5_generation_accuracy,

        "Weighted top-6 generation accuracy":
            weighted_top6_generation_accuracy,

        "Weighted top-7 generation accuracy":
            weighted_top7_generation_accuracy,

        "Weighted top-8 generation accuracy":
            weighted_top8_generation_accuracy,

        "Weighted top-9 generation accuracy":
            weighted_top9_generation_accuracy,

        "Weighted top-10 generation accuracy":
            weighted_top10_generation_accuracy,
    }