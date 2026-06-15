from typing import Dict, Optional


# Returns the list of values for a given property among entity's claims.
def get_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = [claim['mainsnak']['datavalue']['value']['id'] for claim in claims[pid]]
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None


# Returns the list of values for a given property among entity's monolingual claims.
def get_monolingual_claims(claims: Dict, pid: str) -> Optional[list[str]]:
    try:
        result = {claim['mainsnak']['datavalue']['value']['language']: claim['mainsnak']['datavalue']['value']['text']
                  for claim in claims[pid]}
        return result if result else None
    except (KeyError, IndexError, TypeError):
        return None
