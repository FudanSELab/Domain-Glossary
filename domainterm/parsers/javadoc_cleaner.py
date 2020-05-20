#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from bs4 import BeautifulSoup

from .base_cleaner import BaseCleaner

class JavaDocCleaner(BaseCleaner):
    NAME = "Javadoc Cleaner"

    def __init__(self, **cfg):
        super(JavaDocCleaner, self).__init__(**cfg)

    def clean(self, html):
        soup = BeautifulSoup(html, "lxml")
        strings = []
        for div in soup.select(".block"):
            string = div.get_text().strip()
            if len(string) > 0 and string[-1] not in set(".?!"):
                string = string + "."
            strings.append(string)
        text = " ".join(strings)
        sents = [sent for sent in self.split_text(text) if self.check_sentence(sent)]
        return sents
