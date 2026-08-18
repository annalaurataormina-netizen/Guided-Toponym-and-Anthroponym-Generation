import editdistance


def evaluate_novelty_and_diversity(generated_per_language, train_names, culture_counts):
    train_names_per_culture = {}

    for name, language in train_names:
        train_names_per_culture.setdefault(language, set()).add(name)

    novelty_scores = {}
    unique_rates = {}
    near_duplicate_rates = {}

    threshold = 0.25

    for language, generated_names in generated_per_language.items():

        # Training names for this culture
        culture_train_names = train_names_per_culture.get(
            language,
            set()
        )

        novel_names = [
            name
            for name in generated_names
            if name not in culture_train_names
        ]

        novelty_scores[language] = (
            len(novel_names) / len(generated_names)
        )

        unique_rates[language] = (
            len(set(generated_names)) / len(generated_names)
        )

        duplicates = 0
        pairs = 0

        for i in range(len(generated_names)):
            for j in range(i + 1, len(generated_names)):

                if len(generated_names[i]) == 0 and len(generated_names[j]) == 0:
                    continue

                distance = (
                    editdistance.eval(
                        generated_names[i],
                        generated_names[j]
                    )
                    / max(
                        len(generated_names[i]),
                        len(generated_names[j])
                    )
                )

                if distance <= threshold:
                    duplicates += 1

                pairs += 1

        near_duplicate_rates[language] = duplicates / pairs

    languages = [
        language
        for language in generated_per_language
        if language in culture_counts
    ]

    avg_novelty = sum(
        novelty_scores[language]
        for language in languages
    ) / len(languages)

    avg_unique_rate = sum(
        unique_rates[language]
        for language in languages
    ) / len(languages)

    avg_near_duplicate_rate = sum(
        near_duplicate_rates[language]
        for language in languages
    ) / len(languages)

    total_weight = sum(
        culture_counts[language]
        for language in languages
    )

    weighted_novelty = sum(
        novelty_scores[language] * culture_counts[language]
        for language in languages
    ) / total_weight

    weighted_unique_rate = sum(
        unique_rates[language] * culture_counts[language]
        for language in languages
    ) / total_weight

    weighted_near_duplicate_rate = sum(
        near_duplicate_rates[language] * culture_counts[language]
        for language in languages
    ) / total_weight

    all_generated_names = [
        name
        for generated_names in generated_per_language.values()
        for name in generated_names
    ]

    culture_agnostic_exact_duplicate_rate = (
            1 - len(set(all_generated_names)) / len(all_generated_names)
    )

    return {
        "Exact novelty": avg_novelty,
        "Weighted exact novelty": weighted_novelty,

        "Exact diversity": avg_unique_rate,
        "Weighted exact diversity": weighted_unique_rate,

        "Duplicate rate": avg_near_duplicate_rate,
        "Weighted duplicate rate": weighted_near_duplicate_rate,

        "Duplicate rate (culture-agnostic)": culture_agnostic_exact_duplicate_rate,
    }