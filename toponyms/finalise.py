import gzip
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_romanised, get_countries_names, split_diacritics, get_country_languages

MIN_LENGTH_THRESHOLD = 3
MAX_LENGTH_THRESHOLD = 50

COUNTRY_LANGUAGES = get_country_languages()

# Languages that are not really spoken nowadays.
EXCLUDED_EXACT = {
    'Uncoded languages', 'Multiple languages', 'Australian languages',
    'Mycenaean Greek', 'Ancient Greek', 'Sumerian', 'Akkadian',
    'Elamite', 'Phoenician', 'Old Norse', 'Old English', 'Old French',
    'Old Turkish', 'Church Slavic', 'Ancient Hebrew', 'Pali', 'Latin', 'Aramaic',
    'Sanskrit', 'Ancient Egyptian', 'Literary Chinese', 'Ottoman Turkish', 'Church Slavic',
    'Prussian', 'Taino', 'Wyandot', 'Berber languages', 'Western Abnaki', 'Koyukon', 'Pite Sami'
}
# Below have names for less than 200 toponyms.
EXCLUDED_SMALL = {
    'Aleut', 'Navajo', 'Corsican', 'Avaric', 'Siberian Tatar', 'Atikamekw', 'Nahuatl languages',
    'Yoruba', 'Veps', 'Russia Buriat', 'Tahitian', 'Komi-Permyak', 'Pampanga', 'Mazanderani',
    'Uyghur', 'Kara-Kalpak', 'Cebuano', 'Chamorro', 'Paiwan', 'Egyptian Arabic', 'Tachelhit',
    'Picard', 'Haida', 'Lezghian', 'Latgalian', 'Ladin', 'Mari',
    'Antigua and Barbuda Creole English', 'Papiamento', 'Mingrelian', 'Buriat', 'Manx',
    'Moroccan Arabic', 'Northern Hindko', 'Cree', 'Wolaytta', 'Livvi', 'Tyap', 'Tok Pisin',
    'Chipewyan', 'Central Okinawan', 'Northern Frisian', 'Ewe', 'Fiji Hindi', 'Wyandot',
    'Kashmiri', 'Northern Sotho', 'Gagauz', 'Tetum', 'Kildin Sami', 'Nogai', 'Min Dong Chinese',
    'Bunun', 'Saraiki', 'Livonian', 'Dimli (individual language)', 'Neapolitan', 'Zeelandic',
    'Taino', 'Akan', 'Nama', 'Magahi', 'Kabuverdianu', 'Gheg Albanian', 'Hiligaynon', 'Rusyn',
    'Nigerian Pidgin', 'Siksiká', 'Oromo', 'Tausug', 'Talysh', 'Central Atlas Tamazight',
    'Central Kurdish', 'Lakota', 'Northern Tutchone', 'Twi', 'Lower Silesian', 'Ancient Egyptian',
    'Chavacano', 'Kinaray-a', 'Maithili', 'Lillooet', 'Mohawk', 'Silesian', 'Batak Toba',
    'Soninke', 'Hakka Chinese', "Tohono O'odham", 'Esperanto', 'Abaza', 'Ga', 'Eastern Frisian',
    'Zhuang', 'Omaha-Ponca', 'Rukai', 'Thao', 'Warlpiri', 'Northern East Cree', 'Palauan',
    'Yapese', 'Gilbertese', 'Nenets', 'Eastern Canadian Inuktitut', 'Standard Moroccan Tamazight',
    'Chukot', 'Tsimshian', 'Saaroa', 'Jejueo', 'Kundal Shahi', 'Arawak', 'Brahui', 'Zarma',
    'Herero', 'Cajun French', 'Fon', 'Seneca', 'South Azerbaijani', 'Chilcotin', 'Hunsrik',
    'Okanagan', 'Yiddish', 'Võro', 'Western Balochi', 'Munsee', 'Nheengatu', 'Choctaw',
    'Bishnupriya', 'Amis', 'Dogrib', 'Slave', 'Niuean', 'Pitcairn-Norfolk', 'Iloko', 'Romagnol',
    'Southern Balochi', 'Afar', 'Ume Sami', 'Island Carib', 'Mirandese', 'Shughni', 'Judeo-Tat',
    'Goan Konkani', 'Nyungar', 'Ainu', 'Tornedalen Finnish', 'Prussian', 'Ibibio',
    'Southern Tutchone', 'Torwali', 'Syriac', 'Western Armenian', 'Ganda', 'Susu',
    'Central Siberian Yupik', 'Gilaki', 'Rakhine', 'Shan', 'Ndonga', 'Kikuyu', 'Straits Salish',
    'Romany', 'Mixtepec Mixtec', 'Literary Chinese', 'Vlax Romani', 'Kuvi', 'Tagish',
    'Jamaican Creole English', 'Tumbuka', 'Rajasthani', 'Rapanui', 'Marwari (India)', 'Ingrian',
    'Nanai', 'Kulon-Pazeh', 'Colognian', 'Sranan Tongo', 'Hassaniyya', 'Pular', 'Istro Romanian',
    'Louisiana Creole', 'Kanakanabu', 'Rutul', 'Betawi', 'Zuni', 'Dargwa', 'Min Nan Chinese', 'Faroese', 'Jèrriais',
    'Tongan', 'North Ndebele', 'Lower Sorbian', 'Chechen', 'Sundanese', 'Skolt Sami', 'Marshallese', 'Yakut', 'Innu-aimun',
    'Welsh', 'Sicilian', 'Tuvinian', 'Low German', 'Aragonese', 'Samoan', 'Adyghe', 'Karelian', 'Tlingit', 'Dagbani',
    'Central Yupik', "Mi'kmaw", 'Sardinian', 'Scots', 'Filipino', 'Kashubian', 'Bislama', 'Komi', 'Moksha', 'Cornish',
    'Udmurt', 'Kalmyk', 'Friulian', 'Limburgish', 'Inupiaq', 'Walloon', 'Kabardian', 'Western Mari', 'Piedmontese',
    'West Flemish', 'Western Panjabi', 'Breton', 'Wallisian', 'Algonquin', 'Western Frisian', 'Arpitan', 'Karachay-Balkar',
    'Asturian', 'Khakas', 'Hawaiian', 'Lombard', 'Ingush', 'Kumyk', 'Algerian Arabic', 'Lak', 'Eastern Mari', 'Javanese', 'Bavarian',
    'Nauru', 'Erzya', 'Ojibwa', 'Venetian', 'Inuktitut', 'Ligurian', 'Cantonese', 'Southern Altai', 'Igbo', 'Carolinian'
}
EXCLUDED_PARTIAL = {'Unknown'}


def main():
    counter = 0

    character_counter = Counter()

    breakdown_by_language = Counter()
    breakdown_by_country = Counter()
    breakdown_by_length = Counter()

    excluded_characters = 0
    excluded_length = 0
    excluded_language = 0

    with (gzip.open('/vol/bitbucket/at2225/toponyms_cleaned.jsonl.gz', 'rt') as input):
        with gzip.open('/vol/bitbucket/at2225/toponyms_final.jsonl.gz', 'wt') as output:

            for line in input:

                entity = json.loads(line)

                # Skip entities without names.
                if not entity['name']:
                    continue

                for language, name in entity['name'].items():

                    if language in ('Swiss German', 'Kven Finnish','Norwegian Nynorsk', 'Norwegian Bokmål'):
                        continue

                    # Only keep toponyms whose country is a current sovereign state
                    if not any(c in COUNTRY_LANGUAGES for c in entity['country']):
                        continue

                    if language in EXCLUDED_EXACT or language in EXCLUDED_SMALL or any(
                            excl in language for excl in EXCLUDED_PARTIAL):
                        excluded_language += 1
                        continue

                    # Get the romanised version of the name.
                    name_romanised = get_romanised(name['name'])

                    if not name_romanised:
                        excluded_characters += 1
                        continue

                    length = len(name_romanised)

                    if length < MIN_LENGTH_THRESHOLD or length > MAX_LENGTH_THRESHOLD:
                        excluded_length += 1
                        continue

                    name_romanised = split_diacritics(name_romanised)

                    # Ignore rare diacritics and combining characters that appeared when splitting
                    # the diacritics from the underlying character.
                    name_romanised = name_romanised.replace('̍', '')
                    name_romanised = name_romanised.replace('̭', '')
                    name_romanised = name_romanised.replace('̑', '')
                    name_romanised = name_romanised.replace('̓', '')
                    name_romanised = name_romanised.replace('̟', '')

                    if 'ƛ' in name_romanised:
                        excluded_characters += 1
                        continue

                    toponym = {
                        'name_romanised': name_romanised,
                        'name': name['name'],
                        'language': language,
                        'language_code': name['code'],
                        'id': entity['id'],
                        'type': entity['type'],
                        'country': [country for country in entity['country'] if country in COUNTRY_LANGUAGES],
                    }

                    output.write(json.dumps(toponym) + '\n')

                    counter += 1

                    character_counter.update(name_romanised)
                    breakdown_by_language.update([language])
                    breakdown_by_length[length] += 1

                    for country in toponym['country']:
                        breakdown_by_country.update([country])

    countries_id_names = get_countries_names(list(breakdown_by_country.keys()))

    print('# of toponyms: ', counter, '\n')

    print('Character occurrences (romanised): ', character_counter, '\n')

    print('Breakdown by language: ', breakdown_by_language, '\n')
    print('Breakdown by country: ', {countries_id_names.get(k, k): v for k, v in breakdown_by_country.items()}, '\n')
    print('Breakdown by length (romanised): ', breakdown_by_length, '\n')

    print('# of toponyms excluded due to characters: ', excluded_characters, '\n')
    print('# of toponyms excluded due to length: ', excluded_length, '\n')
    print('# of toponyms excluded due to language: ', excluded_language, '\n')


if __name__ == '__main__':
    main()
