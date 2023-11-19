"""Define objects"""
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Optional

from .utils import convert_token_char_map

class DictLike:
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

@dataclass
class DocTokens(DictLike):
    """Tokenized document"""
    doc_id: str
    doc_token_ids: List[int]
    token_to_char: List[Tuple[int, int]]
    doc_text: Optional[str] = None

    def __post_init__(self):
        if self.doc_text is not None:
            self.char_to_token = convert_token_char_map(self.token_to_char, len(self.doc_text))
    
    