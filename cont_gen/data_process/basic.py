"""Define objects"""
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Optional, Dict, Any

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

@dataclass
class QA_Feature(DictLike):
    """
    Context window + question + one answer and inputs to models
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    seq_len: int # non-pad token number
    paragraph_len: int # length of context window
    context_offset: int # start position of context in the input_ids
    span_start: int # answer span start token position in the document
    p_mask: List[int] # 1 for tokens cannot be answers
    cls_index: int # used for impossible span
    start_position: int # answer span start token position in the input_ids
    end_position: int # answer span end token position in the input_ids
    is_impossible: bool # whether the window has answer span
    qa_id: str # {title}_{clause} or {title}_{clause}_{answer_id}
    example_index: int

# Objects for GenQA
class ParaQA(DictLike):
    clause_id: int
    clause_name: str
    reloc_spans: List[Tuple[int, int]] # answer span in paragraph ori_text
    answer_spans: List[Tuple[int, int]] # answer span in paragraph clean_text

class NaturalParagraph(DictLike):
    """
    Natural paragraphs split by linebreaks with answer spans.
    """
    contract_index: int
    para_index: int
    title: str
    offset: int
    length: int # length of original paragraph
    clean_text: str # clean text after removing extra space
    char_map: List[int] # mapping from clean_text to ori_text
    qas: List[ParaQA]

class ParaTokens(DictLike):
    """
    Paragraph tokenization results.
    """
    contract_index: int
    para_index: int
    token_ids: List[int]
    token_to_char: List[Tuple[int, int]]