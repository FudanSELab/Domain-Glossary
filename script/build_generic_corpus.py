#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
from tqdm import tqdm

from domainterm.types import Vocab, Sentence, Token
from domainterm.common import SpacyNLP
from domainterm.parsers import JavaDocCleaner, HTMLCleaner
from domainterm.util.data_loader import DataLoader
from domainterm.util.stopwords import Stopwords


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"
    htmls = DataLoader.html([str(data_dir / "python/html-data"), str(data_dir / "java/html-data")])

    cleaner_factory = {
        "html": HTMLCleaner(),
        "javadoc": JavaDocCleaner()
    }
    html_dict = {"javadoc": set(), "html": set()}

    for html in htmls:
        if html[0] == "javadoc":
            html_dict["javadoc"].add(html[1:])
        else:
            html_dict["html"].add(html[1:])
    print("javadoc: %d" % len(html_dict["javadoc"]))
    print("html: %d" % len(html_dict["html"]))

    sent2sent = dict()
    for name, htmls in html_dict.items():
        if len(htmls) == 0:
            continue
        cleaner = cleaner_factory[name]
        print("{} starts.".format(cleaner.NAME))
        for path, html in tqdm(htmls, desc=f' - ({cleaner.NAME})', leave=False, ascii=True):
            sents = cleaner.clean(html)
            for sent in sents:
                sent = Sentence(sent, sources={path})
                if sent in sent2sent:
                    sent2sent[sent].sources.update(sent.sources)
                else:
                    sent2sent[sent] = sent
    sentences = set(sent2sent.values())
    print("sentences: %d" % len(sentences))

    print("tokenizer starts.")
    vocab = Vocab(stopwords=Stopwords.words("en"))
    nlp = SpacyNLP.get_inst()
    for sent in tqdm(sentences, desc=' - (Tokenization)', leave=False, ascii=True):
        doc = nlp.parse(sent.text)
        tokens = [Token(vocab, t.text, t.lemma_, pos=t.pos_, dep=t.dep_) for t in doc]
        sent.tokens = tokens

    with Path(str(output_dir / "general.corpus")).open("wb") as f:
        pickle.dump((vocab, sentences), f)