import json
from collections import Counter

from sklearn.model_selection import train_test_split

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from CCVAE.evaluate_cultural_coherence import evaluate_cultural_coherence
from CCVAE.evaluate_novelty_and_diversity import evaluate_novelty_and_diversity
from CCVAE.evaluate_pronounceability import evaluate_pronounceability
from CCVAE.test import test
from utils import load_all, normalise


def evaluate(model, lr):
    '''
    for the final evaluation, you wanna be more granular
    you also wanna evaluate ability to generate names from fictional cultures
    '''

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    vocab = CharVocab(ALLOWED_CHARS)

    names = load_all(culture=True)

    names_normalised = [
        [normalise(name), lang]
        for name, lang in names
    ]

    culture_counts = Counter(label for _, label in names_normalised)

    names_normalised = [
        [name, language_to_id[lang]]
        for name, lang in names_normalised
    ]

    # 80/10/10 split of the dataset into train/validation/test (uses same seed as when training the model)
    train_names, temp_names = train_test_split(names_normalised, test_size=0.2, random_state=1996, shuffle=True)
    _, test_names = train_test_split(temp_names, test_size=0.5, random_state=1996, shuffle=True)

    # generate for each language
    generated_per_language = {}

    for language, language_id in language_to_id.items():
        generated_per_language[language] = model.generate(culture=language_id, n=1000, max_length=50, temperature=0.6)

    with open(f"CCVAE/evaluation_results/evaluation_results_lr{lr}_adam.txt", "w") as f:
        f.write(f"Learning rate: {lr}\n")

        f.write("\nTEST\n")
        test_results = test(model, test_names, vocab)
        for k, v in test_results.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nPRONOUNCEABILITY\n")
        pronounceability_results = evaluate_pronounceability(
            generated_per_language,
            culture_counts
        )
        for n, groups in pronounceability_results.items():
            f.write(f"{n}-gram\n")
            for group, metrics in groups.items():
                f.write(
                    f"{group}: "
                    f"Macro avg log-probability: "
                    f"{metrics['Avg log-probability']:.4f} "
                    f"Weighted avg log-probability: "
                    f"{metrics['Weighted avg log-probability']:.4f}\n"
                )

        f.write("\nNOVELTY & DIVERSITY\n")
        novelty_and_diversity_results = evaluate_novelty_and_diversity(generated_per_language, train_names,
                                               culture_counts)
        for k, v in novelty_and_diversity_results.items():
            f.write(f"{k}: {v*100:.2f}%\n")

        f.write("\nCULTURAL COHERENCE (on languages with more than 1000 samples)\n")
        cultural_coherence_results = evaluate_cultural_coherence(generated_per_language, language_to_id, next(model.parameters()).device, vocab,
                                            names_normalised, culture_counts)
        for k, v in cultural_coherence_results.items():
            f.write(f"{k}: {v*100:.2f}%\n")
