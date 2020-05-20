#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path
from nltk.corpus import stopwords

from ..option import Option

with Path(Option.EN_STOPWORDS).open("r", encoding="utf-8") as f:
    EN = {line.strip().lower() for line in f}
with Path(Option.CODE_STOPWORDS).open("r", encoding="utf-8") as f:
    CODE = {line.strip().lower() for line in f}

class Stopwords:
    @staticmethod
    def words(name="en"):
        if name == "en+code":
            return EN | CODE
        elif name == "code":
            return CODE
        else:
            return EN
