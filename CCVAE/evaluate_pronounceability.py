import math

from nGram.nGram import nGram


def evaluate_pronounceability(generated_names_per_language, culture_counts):

    results = {}

    for n in range(2, 5):
        model = nGram(n)
        model.load()

        culture_log_probs = {}
        culture_percentage_inf = {}

        for language, names in generated_names_per_language.items():

            log_probs = [
                model.sequence_log_probability((name, language))
                for name in names
            ]

            finite = [
                x for x in log_probs
                if x != float("-inf")
            ]

            culture_log_probs[language] = (
                sum(finite) / len(finite)
                if finite else float("-inf")
            )

            culture_percentage_inf[language] = (
                100 * (len(log_probs) - len(finite)) / len(log_probs)
            )

        # Only include cultures with a valid weight
        languages = [
            language
            for language in culture_log_probs
            if language in culture_counts
        ]

        # Unweighted averages across cultures
        average_log_probability = sum(
            culture_log_probs[language]
            for language in languages
        ) / len(languages)

        average_percentage_inf = sum(
            culture_percentage_inf[language]
            for language in languages
        ) / len(languages)

        # Weighted averages across cultures
        total_weight = sum(
            culture_counts[language]
            for language in languages
        )

        weighted_average_log_probability = sum(
            culture_log_probs[language] * culture_counts[language]
            for language in languages
        ) / total_weight

        weighted_average_percentage_inf = sum(
            culture_percentage_inf[language] * culture_counts[language]
            for language in languages
        ) / total_weight

        results[n] = {
            "Average log-probability": math.exp(average_log_probability),
            "Percentage of -inf": average_percentage_inf,
            "Weighted average log-probability": math.exp(weighted_average_log_probability),
            "Weighted percentage of -inf": weighted_average_percentage_inf,
        }

    return results
