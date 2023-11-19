"""Shared functions"""

def convert_token_char_map(token_to_char, length):
    """
    Given the token_to_char map, return the char_to_token map.
    For space that not in token, return None.
    Both map is a mapping to span position.
    """
    char_to_token = [[None, None] for _ in range(length)]
    for i, (start, end) in enumerate(token_to_char):
        for j in range(start, end):
            if char_to_token[j][0] is None:
                # start token of the j-th char
                char_to_token[j][0] = i 
            char_to_token[j][1] = i + 1
    return char_to_token