import logging

from functools import partial

from src.data import preprocess_external as pe

logger = logging.getLogger(__name__)


def preprocess(function, sentence1_key, sentence2_key, examples):
    if sentence1_key is not None:
        examples = apply_on_column(function, examples, sentence1_key)
    if sentence2_key is not None:
        examples = apply_on_column(function, examples, sentence2_key)
    return examples


def apply_on_column(function, examples, column):
    examples[column] = [
        function(item)
        for item in examples[column]
    ]
    return examples


def twitter_preprocess(example):
    example = pe.removeURLs(example)
    example = pe.replaceHTMLChar(example)
    example = pe.removeNonAscii(example)
    example = pe.removeNonPrintable(example)
    example = pe.replaceUsernames(example, 'User')
    example = pe.removeRepeatedChars(example)

    return example


PREPROCESSING_FUNCTIONS = {
    'tw1': partial(preprocess, twitter_preprocess),
}
