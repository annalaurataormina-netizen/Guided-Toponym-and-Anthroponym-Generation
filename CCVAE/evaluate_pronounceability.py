from nGram.nGram import nGram


def evaluate_pronounceability(generated_names_per_language, culture_counts):

    results = {}

    for n in range(2, 5):

        model = nGram(n)
        model.load()

        culture_log_probs = {}
        culture_percentage_inf = {}

        # ---------------------------------------------------------
        # Calculate statistics for each culture
        # ---------------------------------------------------------

        for language, names in generated_names_per_language.items():
            log_probs = [
                model.sequence_log_probability((name, language))
                for name in names
            ]

            finite = [
                x for x in log_probs
                if x != float("-inf")
            ]

            # Ignore -inf when calculating the average.
            # If everything is -inf, mark the culture as invalid.
            culture_log_probs[language] = (
                sum(finite) / len(finite)
                if finite
                else None
            )

            # Still report how many scores were -inf.
            culture_percentage_inf[language] = (
                    100 * (len(log_probs) - len(finite))
                    / len(log_probs)
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

            # Cultures with at least one finite score
            valid_languages = [
                language
                for language in selected_languages
                if culture_log_probs[language] is not None
            ]

            # Average log-probability across valid cultures
            average_log_probability = (
                sum(
                    culture_log_probs[language]
                    for language in valid_languages
                )
                / len(valid_languages)
                if valid_languages
                else None
            )

            # Average % of -inf across all selected cultures
            average_percentage_inf = (
                sum(
                    culture_percentage_inf[language]
                    for language in selected_languages
                )
                / len(selected_languages)
                if selected_languages
                else None
            )

            # Weighted log-probability
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

            # Weighted % of -inf
            total_weight_all = sum(
                culture_counts[language]
                for language in selected_languages
            )

            weighted_average_percentage_inf = (
                sum(
                    culture_percentage_inf[language]
                    * culture_counts[language]
                    for language in selected_languages
                )
                / total_weight_all
                if total_weight_all > 0
                else None
            )

            return {
                "Avg log-probability":
                    average_log_probability,

                "% of -inf":
                    average_percentage_inf,

                "Weighted avg log-probability":
                    weighted_average_log_probability,

                "Weighted % of -inf":
                    weighted_average_percentage_inf,
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