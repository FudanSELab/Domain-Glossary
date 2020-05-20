#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import re
from bs4 import BeautifulSoup

class BaseCleaner(metaclass=ABCMeta):
    NAME = "Cleaner"

    PUNC_TABLE = {ord(zh): ord(en) for zh, en in zip('‘’“”…，。！？【】（）％＃＠＆：',
                                                     '\'\'"".,.!?[]()%#@&:')}

    def __init__(self, **cfg):
        pass

    def __call__(self, **args):
        return self.clean(*args)

    @abstractmethod
    def clean(self, html):
        raise NotImplementedError

    def split_text(self, text):
        text = text.translate(BaseCleaner.PUNC_TABLE)
        text = re.sub(r'\n(.+?[^.?!])\n([A-Z])', r'\n\n\2', text)
        text = re.sub(r'\s+', " ", text.strip())
        text = re.sub(r'([?!.]+) ', r'\1\n', text)
        sentences = set(text.split("\n"))
        return sentences

    def check_sentence(self, sentence, min_len=3, max_len=200):
        if len(sentence) == 0 or not (min_len <= len(sentence.split()) <= max_len):
            return False
        # check Chinese chars
        if any(["\u4e00" <= ch <= "\u9fff" for ch in sentence]):
            return False
        return True

    