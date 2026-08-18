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

        # Cultures for which we have generated names and training counts
        languages = [
            language
            for language in culture_log_probs
            if language in culture_counts
        ]

        def calculate_metrics(selected_languages):

            # Unweighted averages across cultures
            average_log_probability = sum(
                culture_log_probs[language]
                for language in selected_languages
            ) / len(selected_languages)

            average_percentage_inf = sum(
                culture_percentage_inf[language]
                for language in selected_languages
            ) / len(selected_languages)

            # Weighted averages across cultures
            total_weight = sum(
                culture_counts[language]
                for language in selected_languages
            )

            weighted_average_log_probability = sum(
                culture_log_probs[language] * culture_counts[language]
                for language in selected_languages
            ) / total_weight

            weighted_average_percentage_inf = sum(
                culture_percentage_inf[language] * culture_counts[language]
                for language in selected_languages
            ) / total_weight

            return {
                "Avg log-probability": average_log_probability,
                "% of -inf": average_percentage_inf,
                "Weighted avg log-probability": weighted_average_log_probability,
                "Weighted % of -inf": weighted_average_percentage_inf,
            }

        # Define the three groups
        languages_1000 = [
            language
            for language in languages
            if culture_counts[language] >= 1000
        ]

        languages_10000 = [
            language
            for language in languages
            if culture_counts[language] >= 10000
        ]

        # Calculate metrics for each group
        results[n] = {
            "All": calculate_metrics(languages),
            ">=1000": calculate_metrics(languages_1000),
            ">=10000": calculate_metrics(languages_10000),
        }

    return results