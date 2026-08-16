from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

from nGram.nGram import nGram
from utils import load_all, normalise


def test():
    seed = 1996

    # Toponyms and Anthroponyms (name_romanised, label)
    names = load_all(culture=True)

    # Normalise name (split diacritics) and replace language codes with integers
    names_normalised = [
        [normalise(name), lang]
        for name, lang in names
    ]

    # 80/10/10 split of the dataset into train/validation/test
    _, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=seed, shuffle=True)
    _, test_names = train_test_split(temp_names, test_size=0.5, random_state=seed, shuffle=True)

    culture_counts = Counter(
        language
        for _, language in names
    )

    names_per_language = defaultdict(list)

    for name, language in test_names:
        names_per_language[language].append(name)

    results = defaultdict(dict)

    for language, names in names_per_language.items():

        for n in range(2, 4):
            model = nGram(n)
            model.load()
            print(f"N = {n}")

            log_probabilities = [
                model.sequence_log_probability((name, language))
                for name in names
            ]

            num_inf = sum(
                1 for log_prob in log_probabilities
                if log_prob == float("-inf")
            )

            valid_log_probabilities = [
                log_prob
                for log_prob in log_probabilities
                if log_prob != float("-inf")
            ]

            percentage_inf = 100 * num_inf / len(log_probabilities)

            if valid_log_probabilities:
                average_log_probability = (
                        sum(valid_log_probabilities)
                        / len(valid_log_probabilities)
                )
            else:
                average_log_probability = float("-inf")

            results[n][language] = {
                "count": culture_counts[language],
                "percentage_inf": percentage_inf,
                "average_log_probability": average_log_probability,
            }

            print(
                f"Language: {language} | "
                f"Counts: {culture_counts[language]} | "
                f"% -inf: {percentage_inf:.2f}% | "
                f"Average log-probability: {average_log_probability:.4f} | "
                f"Valid: {len(valid_log_probabilities)}/1000"
            )

    thresholds = {
        "All": 0,
        ">=1000": 1000,
        ">=10000": 10000,
    }

    for n in range(2, 4):

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

            macro_inf = sum(
                result["percentage_inf"]
                for result in selected
            ) / len(selected)

            valid_log_probs = [
                result["average_log_probability"]
                for result in selected
                if result["average_log_probability"] is not None
            ]

            if valid_log_probs:
                macro_log_prob = (
                        sum(valid_log_probs)
                        / len(valid_log_probs)
                )
            else:
                macro_log_prob = float("-inf")

            total_weight = sum(
                result["count"]
                for result in selected
            )

            weighted_inf = sum(
                result["percentage_inf"] * result["count"]
                for result in selected
            ) / total_weight

            weighted_log_probs = [
                result
                for result in selected
                if result["average_log_probability"] is not None
            ]

            if weighted_log_probs:
                weighted_log_prob = sum(
                    result["average_log_probability"] * result["count"]
                    for result in weighted_log_probs
                ) / sum(
                    result["count"]
                    for result in weighted_log_probs
                )
            else:
                weighted_log_prob = float("-inf")

            print(
                f"  {group_name}: "
                f"{len(selected)} cultures | "
                f"Macro % -inf: {macro_inf:.2f}% | "
                f"Weighted % -inf: {weighted_inf:.2f}% | "
                f"Macro avg log-P: {macro_log_prob:.4f} | "
                f"Weighted avg log-P: {weighted_log_prob:.4f}"
            )


if __name__ == "__main__":
    test()
