import unicodedata

def main():

    chars = ['Ꮎ']
    for char in chars:
        print(char, unicodedata.category(char), unicodedata.name(char, ''))

if __name__ == '__main__':
    main()