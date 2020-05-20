#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re

from bs4 import BeautifulSoup

from .base_cleaner import BaseCleaner


class HTMLCleaner(BaseCleaner):
    NAME = "HTML Cleaner"

    def __init__(self, **cfg):
        super(HTMLCleaner, self).__init__(**cfg)
        # self.nlp = spacy.load("en")

    def clean(self, html):
        body = BeautifulSoup(html, "lxml").body
        # print(body.prettify())
        scripts = body.findAll("script")
        [script.extract() for script in scripts]
        navs = body.findAll(class_=re.compile(r'.*(nav|Nav|footer).*'))
        [nav.extract() for nav in navs]
        footers = body.findAll("footer")
        [footer.extract() for footer in footers]

        for li in body.findAll("li"):
            string = li.get_text().strip()
            if len(string) > 0 and string[-1] not in set(".?!:;,"):
                string = string + "."
            li.clear()
            li.append(string)
        for h in body.findAll(re.compile(r'h[1-6]')):
            string = h.get_text().strip()
            if len(string) > 0 and string[-1] not in set(".?!:;,"):
                string = string + "."
            h.clear()
            h.append(string)
        for p in body.findAll("p"):
            string = p.get_text().strip()
            if len(string) > 0 and string[-1] not in set(".?!:;,"):
                string = string + "."
            p.clear()
            p.append(string)

        for table in body.findAll("table"):
            table.clear()
            table.append("__TABLE__")
        for img in body.findAll("img"):
            if not img.get("alt") or len(img["alt"]) == 0:
                img_alt = "__IMG__"
            else:
                img_alt = img["alt"]
            img.insert_after(img_alt)
        for code in body.findAll("code"):
            string = code.get_text().strip()
            if len(string.split()) >= 3 or len(string) > 20:
                string = "__CODE__"
            code.clear()
            code.append(string)

        text = body.get_text()
        text = text.strip() + " "
        text = re.sub(r'(https?://.*?)([^a-zA-Z0-9/]?\s)', r'__URL__\2', text)
        sents = [sent for sent in self.split_text(text) if self.check_sentence(sent)]
        return sents