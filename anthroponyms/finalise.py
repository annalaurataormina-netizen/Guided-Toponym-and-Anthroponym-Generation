import gzip
import json
import os
import sys
from collections import Counter

from langcodes import Language

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_romanised, get_countries_names, get_country_languages

from clean import OCCURRENCE_THRESHOLD

MIN_LENGTH_THRESHOLD = 2
MAX_LENGTH_THRESHOLD = 20

COUNTRY_LANGUAGES = get_country_languages()

EXCLUDE_LANGUAGES = [
    # Unknown/uncoded
    "Unknown language",
    "Unknown language [eml]",
    "Unknown language [isv]",
    "Uncoded languages",
    "No linguistic content",

    # Constructed languages
    "Esperanto",
    "Interlingua",
    "Interlingue",
    "Lingua Franca Nova",

    # Liturgical/classical/dead
    "Pali",
    "Avestan",
    "Classical Syriac",
    "Literary Chinese",
    "Latin",
    "Old Norse",
    "Old English",
    "Middle English",
    "Old Japanese",
    "Ancient Greek",
    "Ottoman Turkish",
    "Prussian",

    # Language families
    "Bihari languages",
    "Nahuatl languages",

    # Writing system
    "N'Ko",
    "N’Ko",

    # Small (<20 anthroponyms)
    "Ewe",
    "Dzongkha",
    "Filipino",
    "Corsican",
    "Asturian",
    "Chechen",
    "Atikamekw",
    "Sardinian",
    "Fang",
    "Ga",
    "Latgalian",
    "Narom",
    "Frafra",
    "Hawaiian",
    "Akan",
    "Fanti",
    "Moroccan Arabic",
    "Balinese",
    "Abkhazian",
    "Pampanga",
    "Arpitan",
    "Lower Sorbian",
    "Egyptian Arabic",
    "Western Panjabi",
    "Navajo",
    "Min Nan Chinese",
    "Ligurian",
    "Low German",
    "Adyghe",
    "Erzya",
    "Friulian",
    "Dagbani",
    "Silesian",
    "Basaa",
    "Tahitian",
    "Algerian Arabic",
    "Kinaray-a",
    "Sicilian",
    "Bavarian",
    "Kumyk",
    "Venetian",
    "Mapuche",
    "Kalaallisut",
    "Cebuano",
    "Javanese",
    "Lezghian",
    "Tuvinian",
    "Manx",
    "Avaric",
    "Cajun French",
    "Picard",
    "Batak Mandailing",
    "Skolt Sami",
    "Sranan Tongo",
    "Bini",
    "Fiji Hindi",
    "Northern Frisian",
    "Baoulé",
    "Saterland Frisian",
    "Ibibio",
    "Efik",
    "Votic",
    "Central Bikol",
    "Saraiki",
    "Jamaican Creole English",
    "Waray",
    "Cantonese",
    "Kashmiri",
    "Acehnese",
    "Iloko",
    "Eastern Mari",
    "Lower Silesian",
    "Karachay-Balkar",
    "Rusyn",
    "Northern Sotho",
    "Nigerian Pidgin",
    "Buginese",
    "Southern Dagaare",
    "Uyghur",
    "Aramaic",
    "Ganda",
    "Yakut",
    "Limburgish",
    "Tiv",
    "Hiligaynon",
    "Newari",
    "Plautdietsch",
    "Sundanese",
    "Tibetan",
    "Pangasinan",
    "Tachelhit",
    "Papiamento",
    "Aromanian",
    "South Azerbaijani",
    "Bodo",
    "Karelian",
    "Talysh",
    "Veps",
    "Inari Sami",
    "Komi",
    "Goan Konkani",
    "Tornedalen Finnish",
    "Tuvalu",
    "Standard Moroccan Tamazight",
]


def main():
    counter = 0

    character_counter = Counter()

    breakdown_by_language = Counter()
    breakdown_by_country = Counter()
    breakdown_by_length = Counter()

    excluded_characters = 0
    excluded_length = 0

    with gzip.open('/vol/bitbucket/at2225/anthroponyms_cleaned.jsonl.gz', 'rt') as input:
        with gzip.open('/vol/bitbucket/at2225/anthroponyms_final.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # Skip entities without names.
                if not entity['name']:
                    continue

                for language, name in entity['name'].items():

                    if language in EXCLUDE_LANGUAGES:
                        continue

                    # Get the romanised version of the name.
                    name_romanised = get_romanised(name['name'])

                    if not name_romanised:
                        excluded_characters += 1
                        continue

                    if len(name_romanised) < MIN_LENGTH_THRESHOLD or len(name_romanised) > MAX_LENGTH_THRESHOLD:
                        excluded_length += 1
                        continue

                    length = len(name_romanised)

                    # Rare characters (< 5 occurrences in the database)
                    # For rare characters, ignoring diacritics and combining characters and
                    # leaving all those in.
                    name_romanised = name_romanised.replace('ǝ', 'e').replace('ɔ', 'o').replace('ŋ', 'ng')

                    countries_of_birth = entity.get('occurrences', {}).get('country_of_birth', {})

                    countries = [
                        c for c in countries_of_birth.keys()
                        if countries_of_birth[c] >= OCCURRENCE_THRESHOLD and c != 'null' and c is not None
                           and c in COUNTRY_LANGUAGES
                    ]

                    languages_countries_of_birth = []

                    if language == 'Multiple languages':
                        languages_countries_of_birth = [
                            l for c in countries_of_birth
                            if c in COUNTRY_LANGUAGES
                            for l in COUNTRY_LANGUAGES[c]
                        ]

                    if languages_countries_of_birth:
                        for l in languages_countries_of_birth:
                            if Language.get(l).language_name() in entity['name'].keys():
                                continue
                            anthroponym = {
                                'name_romanised': name_romanised,
                                'name': name['name'],
                                'language': Language.get(l).language_name(),
                                'language_code': l,
                                'id': entity['id'],
                                'type': entity['type'],
                                'country': countries,
                            }
                            output.write(json.dumps(anthroponym) + '\n')
                            counter += 1
                            character_counter.update(name_romanised)
                            breakdown_by_language.update([Language.get(l).language_name()])
                            breakdown_by_length[length] += 1

                            for country in anthroponym['country']:
                                breakdown_by_country.update([country])

                    elif language != 'Multiple languages':
                        anthroponym = {
                            'name_romanised': name_romanised,
                            'name': name['name'],
                            'language': language,
                            'language_code': name['code'],
                            'id': entity['id'],
                            'type': entity['type'],
                            'country': countries,
                        }
                        output.write(json.dumps(anthroponym) + '\n')
                        counter += 1
                        character_counter.update(name_romanised)
                        breakdown_by_language.update([language])
                        breakdown_by_length[length] += 1

                        for country in anthroponym['country']:
                            breakdown_by_country.update([country])

    countries_id_names = get_countries_names(list(breakdown_by_country.keys()))

    print('# of anthroponyms: ', counter, '\n')

    print('Character occurrences (romanised): ', character_counter, '\n')

    print('Breakdown by language: ', breakdown_by_language, '\n')
    print('Breakdown by country: ', {countries_id_names.get(k, k): v for k, v in breakdown_by_country.items()}, '\n')
    print('Breakdown by length (romanised): ', breakdown_by_length, '\n')

    print('# of anthroponyms excluded due to characters: ', excluded_characters, '\n')
    print('# of anthroponyms excluded due to length: ', excluded_length, '\n')


if __name__ == '__main__':
    main()
