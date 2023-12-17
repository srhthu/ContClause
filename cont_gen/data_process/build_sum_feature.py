"""
Build the features for summarization where all aspects are generated simultaneously.

To save the context length, if one clause is too long, generated the abbreviation (only keep head and tail)

The input is the tokenized document processed by cut_doc.py.

Some issues to consider:
    - Whatif one span has multiple clause types?
        we can merge the clause types
"""

