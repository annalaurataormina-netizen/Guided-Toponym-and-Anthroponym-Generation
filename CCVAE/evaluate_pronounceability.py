from nGram.nGram import nGram


def evaluate_pronounceability(generated_names_per_language, culture_counts):

    results = {}

    for n in range(2, 5):

        model = nGram(n)
        model.load()

        culture_log_probs = {}

        # ---------------------------------------------------------
        # Calculate average log-probability for each culture
        # ---------------------------------------------------------

        for language, names in generated_names_per_language.items():

            log_probs = [
                model.sequence_log_probability((name, language))
                for name in names
            ]

            culture_log_probs[language] = (
                sum(log_probs) / len(log_probs)
                if log_probs
                else None
            )

        # Cultures for which we have generated names
        # and training counts
        languages = [
            language
            for language in culture_log_probs
            if language in culture_counts
        ]

        # ---------------------------------------------------------
        # Calculate group statistics
        # ---------------------------------------------------------

        def calculate_metrics(selected_languages):

            valid_languages = [
                language
                for language in selected_languages
                if culture_log_probs[language] is not None
            ]

            # Macro average across cultures
            average_log_probability = (
                sum(
                    culture_log_probs[language]
                    for language in valid_languages
                )
                / len(valid_languages)
                if valid_languages
                else None
            )

            # Weighted average across cultures,
            # weighted by number of training examples
            total_weight = sum(
                culture_counts[language]
                for language in valid_languages
            )

            weighted_average_log_probability = (
                sum(
                    culture_log_probs[language]
                    * culture_counts[language]
                    for language in valid_languages
                )
                / total_weight
                if total_weight > 0
                else None
            )

            return {
                "Avg log-probability":
                    average_log_probability,

                "Weighted avg log-probability":
                    weighted_average_log_probability,
            }

        # ---------------------------------------------------------
        # Define groups
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Calculate metrics
        # ---------------------------------------------------------

        results[n] = {
            "All": calculate_metrics(languages),
            ">=1000": calculate_metrics(languages_1000),
            ">=10000": calculate_metrics(languages_10000),
        }

    return results