import random

def select():

    names = 'Pandikhai, Lathel Bir Bula, Lad̏ribem Kadi, Cardherka Bin, Redar-ehalabeg, Belher-Lan, Gerkeat-aura, Pan͐rghe Bathalui, Halhimer-Ri, Caisar-ur, Lalu-la Briga, Shushala-Hin, Relag-Pur, Kamraw H, Päd̏ry-e Stia, Lamŋ̏rie Bal Buraka, Manikuar Hon, Yedinaal , Paghlak Cala Aita, Saal̏rnbebai Pung, Kadawheh, Bungart-eware, San̏rybel Hi, Belkur-Ir, Rakhar-era, Betoura H, Dayderaa Kon, Hadh P, Beul̏rnber-Al Horilak, Banhausarur, Mangerar L, Gasd̏rnbe Kalatin, Hundiwa Salla Din, Rad̏ri-Bet Di, Canghar Pan, Bun̏rr-el Gin, Krunharhi, Anterbeh͖͈heri, Geshaad Halshar, Coudrakai Ga, Wedweraihan, Cons-e Cantominak, Kalow-ea Han, Barwalu Hin, Hed̏ryvisti, Jadn̙alet Kellawe He, Rakte Boli A, Balŋ̏rnbew I, Coldera-Oroin, Cebaltaria, Red̏rnb-kas, Hashaure-tig, Koulakuri, Cendi-kalian, Pallyat-Holla, Blakaura-keri, Wan̏rnib Morik, Mugewah-eal Ch, Bawŋater She, Bard-el Hie, Hard̏riba, Kädhals Cielamer, Karsiahemari , Aldakar-Pure, Barnhera Dan, Bolgherhage, Pad-erawala, Betlakar Sarar, Magolŋ-gur, Barbakhai, Gondhipariar, Band̙arek-Dia, Daugaiwar, Cald̏rizen̏rnged, Soujw Paliash, Kebherhapali , Hedhertar-eale, Rul̏rr-e Pard, Sruedarsal-H, Barhu Ban, Jawd̏rn- Re Tali, Gangerak, Gashale-d, Caldur Kallig, Lomrenŋ-Ralla Sila, Tamgar-ust, Mad̏rivenai, Onsȅrnb-e Pala, Godeiga Bant, Teel̏rnbet Pin, Beohrew-e Palii, Allehause, Joguaŋ̏r-Berde Bulanir, Galŋ̏riben Pathi, Holdil Kara Ra, Gërauw Paee, Wad̏ribenata, Boɦ͖trrher Ping, Tandar Shir, Jongur- Mod'

    names = names.split(', ')

    print(names)
    print(len(names))

    random_indices = random.sample(range(100), 10)

    for idx in random_indices:
        print(names[idx])

if __name__ == "__main__":
    select()