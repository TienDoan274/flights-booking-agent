import json
from difflib import get_close_matches
import os
with open(os.path.join('..', 'iata_code', 'region_name.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

vocab = list(data.values())

def match_region(region: str) -> list:
    """
    Finds the top 5 most similar region names from the vocabulary using string similarity.
    
    Args:
    region (str): User's input string to match.
    
    Returns:
    list: A list of up to 5 most similar region names, or an empty list if no match is found.
    """
    # Find close matches with a similarity threshold
    matches_list = get_close_matches(region, vocab, n=1, cutoff=0.6)  # Adjust cutoff if needed
    if matches_list:
        return matches_list[0]
    else:
        return None  # Return the list of matches (can be empty)

# Example usage
'''
user_input = "hong kong"  # Example incorrect input
corrected_regions = match_region(user_input)

if corrected_regions:
    print(corrected_regions)
else:
    print('Nothing')
'''
