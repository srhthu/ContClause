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
    
@dataclass
class QA_Example(DictLike):
    """
    Context with one question and several answers.

    It will be further splited into several features due to model input length limit

    For training examples, there is only one answer as several answers are splited into severl examples.
    """
    doc_id: str
    paragraph_id: int
    clause_id: int
    qa_id: str
    question_text: str
    is_impossible: bool
    answer_spans: List[Tuple[int, int]] # end not included
    answer_texts: List[str]

    def parse_id(self):
        """Split the qa_id into multiple information"""
        title, clause_ = self.qa_id.split('__')
        # assert title == self.doc_id
        if clause_[-1].isdigit():
            # this is a training example
            clause, answer_i = clause_.split('_')
        else:
            clause = clause_
            answer_i = None
        return title, clause, answer_i