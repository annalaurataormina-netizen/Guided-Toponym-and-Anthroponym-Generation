import unicodedata

def main():

    char = 'ƛ'
    print(unicodedata.category(char), unicodedata.name(char, ''))

if __name__ == '__main__':
    main()