import json
import random
import statistics
from collections import Counter

import torch

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from CultureClassifier.CultureClassifier import CultureClassifier
from nGram.nGram import nGram
from utils import load_all, normalise


def fictional_culture():

    # ---------------------------------------------------------
    # Reproducibility / device
    # ---------------------------------------------------------

    seed = 1996
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load culture IDs
    # ---------------------------------------------------------

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    # ---------------------------------------------------------
    # Load names and count cultures
    # ---------------------------------------------------------

    names = load_all(culture=True)

    culture_counts = Counter(
        language
        for _, language in names
    )

    # Cultures used for fictional-culture generation
    min_culture_names = 10000

    eligible_cultures = [
        language
        for language, count in culture_counts.items()
        if count >= min_culture_names
    ]

    print(
        f"Eligible cultures (>={min_culture_names:,} names): "
        f"{len(eligible_cultures)}"
    )

    # ---------------------------------------------------------
    # Experiment parameters
    # ---------------------------------------------------------

    n_pairs = 50
    n_generated = 100

    # ---------------------------------------------------------
    # Model hyperparameters
    # ---------------------------------------------------------

    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    lr = 0.0005
    epochs = 50
    patience = 10
    beta_max = 0.025
    n_epochs_ramp_up = 5
    temperature = 0.1
    lambda_supcon = 0.75
    culture_embed_dim = 64

    vocab = CharVocab(ALLOWED_CHARS)

    # ---------------------------------------------------------
    # Load Conditional VAE
    # ---------------------------------------------------------

    model = ConditionalVAE(
        vocab,
        embed_dim,
        hidden_dim_encoder,
        hidden_dim_decoder,
        num_layers_encoder,
        num_layers_decoder,
        latent_dim,
        num_cultures,
        culture_embed_dim
    )

    model_name = (
        f"CCVAE/models/best_model_conditional_supcon_logits_"
        f"bi_"
        f"bs{batch_size}_"
        f"ed{embed_dim}_"
        f"hde{hidden_dim_encoder}_"
        f"hdd{hidden_dim_decoder}_"
        f"nle{num_layers_encoder}_"
        f"nld{num_layers_decoder}_"
        f"ld{latent_dim}_"
        f"lr{lr}adam_"
        f"ep{epochs}es{patience}_"
        f"cd0.25_"
        f"blf0t{beta_max}o{n_epochs_ramp_up}_"
        f"ced{culture_embed_dim}_"
        f"t{temperature}_"
        f"l{lambda_supcon}.pt"
    )

    print()
    print(f"Model: {model_name}")

    model.load_state_dict(
        torch.load(
            model_name,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Load n-gram models
    # ---------------------------------------------------------

    ngram_models = {}

    for n in (2, 3, 4):
        ngram_models[n] = nGram(n)
        ngram_models[n].load()

    # ---------------------------------------------------------
    # Load culture classifier
    #
    # The classifier was trained using cultures with >=1000
    # samples, so this threshold is intentionally different
    # from the >=10000 threshold used for selecting culture
    # pairs.
    # ---------------------------------------------------------

    classifier_min_samples = 1000

    names_normalised = [
        [normalise(name), language_to_id[language]]
        for name, language in names
    ]

    classifier_culture_counts = Counter(
        label
        for _, label in names_normalised
    )

    names_normalised = [
        x
        for x in names_normalised
        if classifier_culture_counts[x[1]]
        >= classifier_min_samples
    ]

    remaining_cultures = sorted(
        set(
            label
            for _, label in names_normalised
        )
    )

    old_to_new = {
        old: new
        for new, old in enumerate(remaining_cultures)
    }

    classifier_language_to_id = {
        language: old_to_new[label]
        for language, label in language_to_id.items()
        if label in old_to_new
    }

    classifier_id_to_language = {
        value: key
        for key, value in classifier_language_to_id.items()
    }

    num_cultures_filtered = len(
        classifier_language_to_id
    )

    classifier_batch_size = 512
    classifier_embed_dim = 32
    classifier_hidden_dim = 256
    classifier_num_layers = 1
    classifier_lr = 0.0005
    classifier_epochs = 30

    classifier = CultureClassifier(
        vocab,
        classifier_embed_dim,
        classifier_hidden_dim,
        classifier_num_layers,
        num_cultures_filtered
    )

    classifier_name = (
        f"CultureClassifier/models/"
        f"best_model_"
        f"bs{classifier_batch_size}_"
        f"ed{classifier_embed_dim}_"
        f"hd{classifier_hidden_dim}_"
        f"nl{classifier_num_layers}_"
        f"lr{classifier_lr}_"
        f"ep{classifier_epochs}_"
        f"ms{classifier_min_samples}.pt"
    )

    print()
    print(f"Classifier: {classifier_name}")

    classifier.load_state_dict(
        torch.load(
            classifier_name,
            map_location=device
        )
    )

    classifier.to(device)
    classifier.eval()

    # ---------------------------------------------------------
    # Select unique culture pairs
    # ---------------------------------------------------------

    possible_pairs = [
        (culture_a, culture_b)
        for i, culture_a in enumerate(eligible_cultures)
        for culture_b in eligible_cultures[i + 1:]
    ]

    if n_pairs > len(possible_pairs):
        raise ValueError(
            "n_pairs is larger than the number of possible "
            "unique culture pairs."
        )

    culture_pairs = random.sample(
        possible_pairs,
        n_pairs
    )

    print()
    print("=" * 70)
    print("FICTIONAL CULTURE EVALUATION")
    print("=" * 70)

    print(
        f"Culture pairs: {n_pairs}"
    )

    print(
        f"Names generated per pair: {n_generated}"
    )

    print(
        f"Total names generated: "
        f"{n_pairs * n_generated:,}"
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    pair_results = []

    # ---------------------------------------------------------
    # Evaluate every culture pair
    # ---------------------------------------------------------

    with torch.no_grad():

        for pair_number, (culture_a, culture_b) in enumerate(
            culture_pairs,
            start=1
        ):

            print()
            print("=" * 70)
            print(
                f"PAIR {pair_number}/{n_pairs}: "
                f"{culture_a} + {culture_b}"
            )
            print("=" * 70)

            id_a = language_to_id[culture_a]
            id_b = language_to_id[culture_b]

            print(
                f"{culture_a}: "
                f"{culture_counts[culture_a]:,} names"
            )

            print(
                f"{culture_b}: "
                f"{culture_counts[culture_b]:,} names"
            )

            # -------------------------------------------------
            # Midpoint culture embedding
            # -------------------------------------------------

            labels = torch.tensor(
                [id_a, id_b],
                dtype=torch.long,
                device=device
            )

            embeddings = model.decoder.culture_embedding(
                labels
            )

            embedding_a = embeddings[0]
            embedding_b = embeddings[1]

            fictional_embedding = (
                embedding_a + embedding_b
            ) / 2

            # -------------------------------------------------
            # Generate names
            # -------------------------------------------------

            generated = model.generate(
                culture_embedding=fictional_embedding,
                n=n_generated,
                max_length=50,
                temperature=0.6
            )

            # -------------------------------------------------
            # N-GRAM EVALUATION
            # -------------------------------------------------

            pair_ngram_stats = {}

            parent_cultures = {
                culture_a,
                culture_b
            }

            for n in (2, 3, 4):

                ngram = ngram_models[n]

                # ---------------------------------------------
                # Get scores for every generated name under
                # every eligible culture.
                # ---------------------------------------------

                culture_scores = {
                    culture: []
                    for culture in eligible_cultures
                }

                for name in generated:

                    scores = (
                        ngram.sequence_log_probability_per_culture(
                            name
                        )
                    )

                    for culture in eligible_cultures:

                        score = scores.get(
                            culture,
                            float("-inf")
                        )

                        culture_scores[culture].append(
                            score
                        )

                # ---------------------------------------------
                # Average scores by culture
                # ---------------------------------------------

                average_scores = {}

                for culture, scores in culture_scores.items():

                    finite_scores = [
                        score
                        for score in scores
                        if score != float("-inf")
                    ]

                    if finite_scores:

                        average_scores[culture] = (
                            sum(finite_scores)
                            / len(finite_scores)
                        )

                    else:

                        average_scores[culture] = None

                # ---------------------------------------------
                # Parent-culture score
                #
                # Average the average score under A and B.
                # ---------------------------------------------

                parent_scores = [
                    average_scores[culture]
                    for culture in parent_cultures
                    if average_scores[culture] is not None
                ]

                parent_score = (
                    statistics.mean(parent_scores)
                    if parent_scores
                    else None
                )

                # ---------------------------------------------
                # Other-culture score
                #
                # Average across every eligible culture except
                # the two parent cultures.
                # ---------------------------------------------

                other_scores = [
                    score
                    for culture, score in average_scores.items()
                    if (
                        culture not in parent_cultures
                        and score is not None
                    )
                ]

                other_score = (
                    statistics.mean(other_scores)
                    if other_scores
                    else None
                )

                # ---------------------------------------------
                # Parent vs other difference
                #
                # Higher = generated names are more strongly
                # modelled by the parent cultures.
                # ---------------------------------------------

                parent_vs_other_difference = (
                    parent_score - other_score
                    if (
                        parent_score is not None
                        and other_score is not None
                    )
                    else None
                )

                pair_ngram_stats[n] = {
                    "parent_score": parent_score,
                    "other_score": other_score,
                    "parent_vs_other_difference":
                        parent_vs_other_difference,
                    "average_scores": average_scores,
                }

                print()
                print(f"{n}-GRAM")

                if parent_score is not None:
                    print(
                        f"Parent cultures average: "
                        f"{parent_score:.4f}"
                    )
                else:
                    print(
                        "Parent cultures average: N/A"
                    )

                if other_score is not None:
                    print(
                        f"Other cultures average: "
                        f"{other_score:.4f}"
                    )
                else:
                    print(
                        "Other cultures average: N/A"
                    )

                if parent_vs_other_difference is not None:
                    print(
                        f"Parent - other difference: "
                        f"{parent_vs_other_difference:.4f}"
                    )
                else:
                    print(
                        "Parent - other difference: N/A"
                    )

            # -------------------------------------------------
            # N-GRAM RANKING
            # -------------------------------------------------

            # Use the 4-gram ranking here.
            #
            # The ranking is separate from the pronounceability
            # comparison above.
            # -------------------------------------------------

            rank_counts = {
                1: 0,
                2: 0,
                5: 0,
                10: 0,
                20: 0,
            }

            valid_generated = 0

            ngram = ngram_models[4]

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

                valid_generated += 1

                ranked = sorted(
                    valid_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for k in rank_counts:

                    top_k = {
                        culture
                        for culture, _ in ranked[:k]
                    }

                    if parent_cultures & top_k:
                        rank_counts[k] += 1

            print()
            print("4-GRAM PARENT RANKING")

            for k in (1, 2, 5, 10, 20):

                if valid_generated:
                    rate = (
                        rank_counts[k]
                        / valid_generated
                    )
                else:
                    rate = 0.0

                print(
                    f"Parent cultures top-{k}: "
                    f"{rate:.2%}"
                )

            # -------------------------------------------------
            # CULTURE CLASSIFIER
            # -------------------------------------------------

            classifier_predictions = []

            for name in generated:

                sequence = vocab.encode(name)

                sequence = torch.tensor(
                    sequence,
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                length = torch.tensor(
                    [len(sequence[0])],
                    device=device
                )

                logits = classifier(
                    sequence,
                    length
                )

                prediction = logits.argmax(
                    dim=1
                ).item()

                predicted_culture = (
                    classifier_id_to_language[
                        prediction
                    ]
                )

                classifier_predictions.append(
                    predicted_culture
                )

            prediction_counts = Counter(
                classifier_predictions
            )

            parent_predictions = sum(
                prediction_counts[culture]
                for culture in parent_cultures
            )

            # -------------------------------------------------
            # Store pair-level results
            # -------------------------------------------------

            pair_result = {
                "culture_a": culture_a,
                "culture_b": culture_b,

                # N-gram ranking
                "parent_rank_1": (
                    rank_counts[1] / valid_generated
                    if valid_generated
                    else 0.0
                ),
                "parent_rank_2": (
                    rank_counts[2] / valid_generated
                    if valid_generated
                    else 0.0
                ),
                "parent_rank_5": (
                    rank_counts[5] / valid_generated
                    if valid_generated
                    else 0.0
                ),
                "parent_rank_10": (
                    rank_counts[10] / valid_generated
                    if valid_generated
                    else 0.0
                ),
                "parent_rank_20": (
                    rank_counts[20] / valid_generated
                    if valid_generated
                    else 0.0
                ),

                # Classifier
                "classifier_parent_rate": (
                    parent_predictions / n_generated
                ),
                "classifier_culture_a_rate": (
                    prediction_counts[culture_a]
                    / n_generated
                ),
                "classifier_culture_b_rate": (
                    prediction_counts[culture_b]
                    / n_generated
                ),

                # N-gram pronounceability
                "ngram_stats": pair_ngram_stats,
            }

            pair_results.append(
                pair_result
            )

    # =========================================================
    # AGGREGATED RESULTS
    # =========================================================

    print()
    print("=" * 70)
    print("AGGREGATED RESULTS ACROSS CULTURE PAIRS")
    print("=" * 70)

    # ---------------------------------------------------------
    # N-GRAM PARENT VS OTHER CULTURES
    # ---------------------------------------------------------

    print()
    print("N-GRAM PRONOUNCEABILITY")
    print()
    print(
        "Positive parent-vs-other differences indicate that "
        "generated names have higher n-gram log-probability "
        "under the parent cultures than under other cultures."
    )

    for n in (2, 3, 4):

        parent_scores = [
            result["ngram_stats"][n]["parent_score"]
            for result in pair_results
            if result["ngram_stats"][n]["parent_score"]
            is not None
        ]

        other_scores = [
            result["ngram_stats"][n]["other_score"]
            for result in pair_results
            if result["ngram_stats"][n]["other_score"]
            is not None
        ]

        differences = [
            result["ngram_stats"][n][
                "parent_vs_other_difference"
            ]
            for result in pair_results
            if result["ngram_stats"][n][
                "parent_vs_other_difference"
            ] is not None
        ]

        print()
        print(f"{n}-GRAM")

        print(
            f"Parent cultures: "
            f"{statistics.mean(parent_scores):.4f} "
            f"+/- "
            f"{statistics.stdev(parent_scores):.4f}"
        )

        print(
            f"Other cultures: "
            f"{statistics.mean(other_scores):.4f} "
            f"+/- "
            f"{statistics.stdev(other_scores):.4f}"
        )

        print(
            f"Parent - other: "
            f"{statistics.mean(differences):.4f} "
            f"+/- "
            f"{statistics.stdev(differences):.4f}"
        )

    # ---------------------------------------------------------
    # N-GRAM RANKING
    # ---------------------------------------------------------

    print()
    print("N-GRAM PARENT-CULTURE RANKING")
    print()

    for k in (1, 2, 5, 10, 20):

        values = [
            result[f"parent_rank_{k}"]
            for result in pair_results
        ]

        print(
            f"Parent cultures top-{k}: "
            f"{statistics.mean(values):.2%} "
            f"+/- "
            f"{statistics.stdev(values):.2%}"
        )

    # ---------------------------------------------------------
    # CULTURE CLASSIFIER
    # ---------------------------------------------------------

    print()
    print("CULTURE CLASSIFIER")
    print()

    classifier_parent_rates = [
        result["classifier_parent_rate"]
        for result in pair_results
    ]

    classifier_a_rates = [
        result["classifier_culture_a_rate"]
        for result in pair_results
    ]

    classifier_b_rates = [
        result["classifier_culture_b_rate"]
        for result in pair_results
    ]

    print(
        f"Parent cultures: "
        f"{statistics.mean(classifier_parent_rates):.2%} "
        f"+/- "
        f"{statistics.stdev(classifier_parent_rates):.2%}"
    )

    print(
        f"Culture A: "
        f"{statistics.mean(classifier_a_rates):.2%} "
        f"+/- "
        f"{statistics.stdev(classifier_a_rates):.2%}"
    )

    print(
        f"Culture B: "
        f"{statistics.mean(classifier_b_rates):.2%} "
        f"+/- "
        f"{statistics.stdev(classifier_b_rates):.2%}"
    )

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        f"Evaluated {n_pairs} unique culture pairs, "
        f"with {n_generated} generated names per pair."
    )

    print(
        "For each pair, the midpoint of the two culture "
        "embeddings was used to generate the fictional culture."
    )

    print(
        "N-gram pronounceability compares the average "
        "log-probability under the two parent cultures "
        "against the average under all other eligible cultures."
    )

    print(
        "The classifier measures how often generated names "
        "are classified as either parent culture."
    )


if __name__ == "__main__":
    fictional_culture()