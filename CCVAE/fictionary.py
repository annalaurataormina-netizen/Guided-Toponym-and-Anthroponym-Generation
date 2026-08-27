# -------------------------------------------------
# N-GRAM RANKING
# -------------------------------------------------

# Calculate parent-culture rankings separately for
# the 2-, 3-, and 4-gram models.
#
# For each generated name, cultures are ranked by
# n-gram log-probability. The result records whether
# either parent culture appears in the top-k cultures.
# -------------------------------------------------

ngram_rank_counts = {
    n: {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        10: 0,
    }
    for n in (2, 3, 4)
}

ngram_valid_generated = {
    n: 0
    for n in (2, 3, 4)
}

for n in (2, 3, 4):

    ngram = ngram_models[n]

    for name in generated:

        scores = (
            ngram.sequence_log_probability_per_culture(
                name
            )
        )

        valid_scores = {
            culture: score
            for culture, score in scores.items()
            if (
                    culture in eligible_cultures
                    and score != float("-inf")
            )
        }

        if not valid_scores:
            continue

        ngram_valid_generated[n] += 1

        ranked = sorted(
            valid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for k in ngram_rank_counts[n]:

            top_k = {
                culture
                for culture, _ in ranked[:k]
            }

            if parent_cultures & top_k:
                ngram_rank_counts[n][k] += 1

print()
print("N-GRAM PARENT RANKING")

for n in (2, 3, 4):

    print()
    print(f"{n}-GRAM")

    for k in (1, 2, 3, 4, 5, 10):

        if ngram_valid_generated[n]:

            rate = (
                    ngram_rank_counts[n][k]
                    / ngram_valid_generated[n]
            )

        else:

            rate = 0.0

        print(
            f"Parent cultures top-{k}: "
            f"{rate:.2%}"
        )