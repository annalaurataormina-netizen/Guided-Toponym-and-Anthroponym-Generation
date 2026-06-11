'''
def is_country(entity):
    claims = entity.get('claims', {})
    p31 = claims.get('P31', [])
    for claim in p31:
        try:
            val = claim['mainsnak']['datavalue']['value']['id']
            # country
            if val == 'Q6256':
                return True
            # sovereign state
            if val == 'Q3624078':
                return True
            # state
            if val == 'Q7275':
                return True
            # constituent country
            if val == 'Q1763527':
                return True
            # state with limited recognition
            if val == 'Q15634554':
                return True
        except (KeyError, TypeError):
            pass
    return False

def is_river_or_lake(entity):
    claims = entity.get('claims', {})
    p31 = claims.get('P31', [])
    for claim in p31:
        try:
            val = claim['mainsnak']['datavalue']['value']['id']
            # river
            if val == '4022':
                return True
            # lake
            if val == 'Q23397':
                return True
        except (KeyError, TypeError):
            pass
    return False

def is_city_or_town(entity):
    claims = entity.get('claims', {})
    p31 = claims.get('P31', [])
    for claim in p31:
        try:
            val = claim['mainsnak']['datavalue']['value']['id']
            # city or capital city
            if val == 'Q515' or val == 'Q5119':
                return True
            # town
            if val == 'Q3957':
                return True
            # village
            if val == 'Q532':
                return True
            # human settlement
            if val == 'Q486972':
                return True
            # borough
            if val == 'Q188509':
                return True
            # municipality
            if val == 'Q15284':
                return True
        except (KeyError, TypeError):
            pass
    return False

def is_mountain(entity):
    claims = entity.get('claims', {})
    p31 = claims.get('P31', [])
    for claim in p31:
        try:
            val = claim['mainsnak']['datavalue']['value']['id']
            # river
            if val == '4022':
                return True
            # lake
            if val == 'Q23397':
                return True
        except (KeyError, TypeError):
            pass
    return False
'''