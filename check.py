from collections import defaultdict

from utils import load_all, load_anthroponyms, load_toponyms


def check():
    names = load_anthroponyms(culture=True)
    lang_dict = defaultdict(int)
    print("Total:", len(names))
    print("Languages:", len(set(language for _, language in names)))
    for n in names:
        lang_dict[n[1]] += 1
    print("Languages > 100:", len([language for language, count in lang_dict.items() if count > 100]))
    print("Languages > 1000:", len([language for language, count in lang_dict.items() if count > 1000]))
    print("Languages > 10000:", len([language for language, count in lang_dict.items() if count > 10000]))
    names = load_toponyms(culture=True)
    lang_dict = defaultdict(int)
    print("Total:", len(names))
    print("Languages:", len(set(language for _, language in names)))
    for n in names:
        lang_dict[n[1]] += 1
    print("Languages > 100:", len([language for language, count in lang_dict.items() if count > 100]))
    print("Languages > 1000:", len([language for language, count in lang_dict.items() if count > 1000]))
    print("Languages > 10000:", len([language for language, count in lang_dict.items() if count > 10000]))



if __name__ == "__main__":
    check()
