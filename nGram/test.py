import json

from nGram.nGram import nGram


def test():
    for n in range(2, 5):
        model = nGram(n)
        model.load()
        with open("language_to_id.json", "r") as f:
            language_to_id = json.load(f)
        culture_id = language_to_id["Italian"]
        generated = model.generate(culture=culture_id, n=50, max_length=50)
        for name in generated:
            print(f"Name: {name}, Log-probability: {model.log_probability}")

if __name__ == "main":
    test()