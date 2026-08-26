from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

from nGram.nGram import nGram
from utils import load_all, normalise


def test():
    seed = 1996

    # ---------------------------------------------------------
    # Load and normalise dataset
    # ---------------------------------------------------------

    # Toponyms and Anthroponyms: (name_romanised, language)
    names = load_all(culture=True)

    names_normalised = [
        [normalise(name), language]
        for name, language in names
    ]

    # ---------------------------------------------------------
    # 80/10/10 split
    # ---------------------------------------------------------

    _, temp_names = train_test_split(
        names_normalised,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    _, test_names = train_test_split(
        temp_names,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )

    # Number of examples per language in the full dataset
    culture_counts = Counter(
        language
        for _, language in names
    )

    # Group test names by language
    names_per_language = defaultdict(list)

    for name, language in test_names:
        names_per_language[language].append(name)

    # ---------------------------------------------------------
    # Evaluate test set for each n-gram model
    # ---------------------------------------------------------

    results = defaultdict(dict)

    for language, language_names in names_per_language.items():

        for n in range(2, 5):

            model = nGram(n)
            model.load()

            log_probabilities = [
                model.sequence_log_probability((name, language))
                for name in language_names
            ]

            # Number of sequences that could not be scored
            num_none = sum(
                1
                for log_prob in log_probabilities
                if log_prob is None
            )

            # Keep only sequences with a valid log-probability
            valid_log_probabilities = [
                log_prob
                for log_prob in log_probabilities
                if log_prob is not None
            ]

            # Percentage of unscorable sequences
            percentage_none = (
                100 * num_none / len(log_probabilities)
                if log_probabilities
                else 0
            )

            # Average log-probability for this language
            if valid_log_probabilities:
                average_log_probability = (
                    sum(valid_log_probabilities)
                    / len(valid_log_probabilities)
                )
            else:
                average_log_probability = None

            results[n][language] = {
                "count": culture_counts[language],
                "percentage_none": percentage_none,
                "average_log_probability": average_log_probability,
            }

            print(
                f"N = {n} | "
                f"Language: {language} | "
                f"Count: {culture_counts[language]} | "
                f"% None: {percentage_none:.2f}% | "
                f"Average log-probability: "
                f"{average_log_probability:.4f}"
                if average_log_probability is not None
                else
                f"N = {n} | "
                f"Language: {language} | "
                f"Count: {culture_counts[language]} | "
                f"% None: {percentage_none:.2f}% | "
                f"Average log-probability: None"
            )

    # ---------------------------------------------------------
    # Define language-frequency groups
    # ---------------------------------------------------------

    thresholds = {
        "All": 0,
        ">=1000": 1000,
        ">=10000": 10000,
    }

    # ---------------------------------------------------------
    # Calculate macro and weighted statistics
    # ---------------------------------------------------------

    for n in range(2, 5):

        print(f"\n{'=' * 70}")
        print(f"N = {n}")
        print(f"{'=' * 70}")

        culture_results = results[n]

        for group_name, min_count in thresholds.items():

            selected = [
                result
                for result in culture_results.values()
                if result["count"] >= min_count
            ]

            if not selected:
                continue

            # -------------------------------------------------
            # Macro average percentage None
            # -------------------------------------------------

            macro_none = (
                sum(
                    result["percentage_none"]
                    for result in selected
                )
                / len(selected)
            )

            # -------------------------------------------------
            # Macro average log-probability
            # -------------------------------------------------

            valid_results = [
                result
                for result in selected
                if result["average_log_probability"] is not None
            ]

            if valid_results:
                macro_log_prob = (
                    sum(
                        result["average_log_probability"]
                        for result in valid_results
                    )
                    / len(valid_results)
                )
            else:
                macro_log_prob = None

            # -------------------------------------------------
            # Weighted average percentage None
            # -------------------------------------------------

            total_weight = sum(
                result["count"]
                for result in selected
            )

            weighted_none = (
                sum(
                    result["percentage_none"]
                    * result["count"]
                    for result in selected
                )
                / total_weight
                if total_weight > 0
                else None
            )

            # -------------------------------------------------
            # Weighted average log-probability
            # -------------------------------------------------

            if valid_results:

                weighted_log_prob = (
                    sum(
                        result["average_log_probability"]
                        * result["count"]
                        for result in valid_results
                    )
                    / sum(
                        result["count"]
                        for result in valid_results
                    )
                )

            else:
                weighted_log_prob = None

            # -------------------------------------------------
            # Print results
            # -------------------------------------------------

            macro_log_prob_string = (
                f"{macro_log_prob:.4f}"
                if macro_log_prob is not None
                else "None"
            )

            weighted_log_prob_string = (
                f"{weighted_log_prob:.4f}"
                if weighted_log_prob is not None
                else "None"
            )

            weighted_none_string = (
                f"{weighted_none:.2f}"
                if weighted_none is not None
                else "None"
            )

            print(
                f"{group_name}: "
                f"{len(selected)} languages | "
                f"Macro avg log-probability: "
                f"{macro_log_prob_string} | "
                f"Weighted avg log-probability: "
                f"{weighted_log_prob_string} | "
                f"Macro % None: {macro_none:.2f}% | "
                f"Weighted % None: {weighted_none_string}%"
            )


if __name__ == "__main__":
    test()