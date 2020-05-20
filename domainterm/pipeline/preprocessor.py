#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
import spacy
import re
from pathlib import Path
from tqdm import tqdm

from .pipe import Pipe
from ..common import SpacyNLP
from ..parsers import (
    HTMLCleaner,
    JavaDocCleaner,
    JavaParser,
    PythonParser
)
from ..types import (
    Vocab,
    Token,
    Sentence,
    Corpus,
)
from ..util.stopwords import Stopwords
from ..option import Option


class Preprocessor(Pipe):
    def __init__(self, **cfg):
        super(Preprocessor, self).__init__(**cfg)
        self.cleaner_factory = {
            "html": HTMLCleaner(),
            "javadoc": JavaDocCleaner()
        }
        self.parser_factory = {
            "java": JavaParser(),
            "python": PythonParser()
        }
        self.nlp = SpacyNLP.get_inst()

    def despatch(self, htmls, codes):
        html_dict = {"javadoc": set(), "html": set()}
        code_dict = {"java": set(), "python": set()}

        for html in htmls:
            if html[0] == "javadoc":
                html_dict["javadoc"].add(html[1:])
            else:
                html_dict["html"].add(html[1:])
        print("javadoc: %d" % len(html_dict["javadoc"]))
        print("html: %d" % len(html_dict["html"]))

        for code in codes:
            if code[0] == "java":
                code_dict["java"].add(code[1:])
            else:
                code_dict["python"].add(code[1:])
        print("java code: %d" % len(code_dict["java"]))
        print("python code: %d" % len(code_dict["python"]))
        return html_dict, code_dict

    def tokenize(self, corpus:Corpus):
        print("tokenizer starts.")
        for sent in tqdm(corpus.sentences, desc=' - (Tokenizer)', leave=False, ascii=True):
            doc = self.nlp.parse(sent.text)
            tokens = [Token(corpus.vocab, t.text, t.lemma_, pos=t.pos_, dep=t.dep_) for t in doc]
            sent.tokens = tokens
            nps = {(np.start, np.end) for np in doc.noun_chunks}
            sent.add_nps(*nps)

    def train_wv(self, corpus:Corpus, window=5, min_count=1, glove_path=Option.GLOVE):
        print("w2v starts.")
        sents = [[str(token) for token in sent] for sent in corpus.sentences]
        model = Word2Vec(size=corpus.vocab.emb_size, window=window, min_count=min_count)
        model.build_vocab(sents)

        # load glove embeddings
        with Path(glove_path).open("r", encoding="utf-8") as f:
            glove = [line.strip() for line in f if " ".join(line.split()[:-corpus.vocab.emb_size]) in corpus.vocab.words]
        temp_file = Path(__file__).parent / "temp.txt"
        with temp_file.open("w", encoding="utf-8") as f:
            f.write("{} {}\n".format(len(glove), corpus.vocab.emb_size))
            f.write("\n".join(glove))
        model.intersect_word2vec_format(str(temp_file), lockf=1.)
        temp_file.unlink()

        model.train(sents, total_examples=model.corpus_count, epochs=model.epochs)
        for word in model.wv.vocab.keys():
            corpus.vocab.embeddings[word] = model.wv[word]

    def process(self, htmls, codes):
        html_dict, code_dict = self.despatch(htmls, codes)
        corpus = Corpus(Vocab(stopwords=Stopwords.words("en")))

        sent2sent = dict()

        for name, htmls in html_dict.items():
            if len(htmls) == 0:
                continue
            cleaner = self.cleaner_factory[name]
            print("{} starts.".format(cleaner.NAME))
            for path, html in tqdm(htmls, desc=f' - ({cleaner.NAME})', leave=False, ascii=True):
                sents = cleaner.clean(html)
                for sent in sents:
                    sent = Sentence(sent, sources={str(Path(path).parts[1])})
                    if sent in sent2sent:
                        sent2sent[sent].sources.update(sent.sources)
                    else:
                        corpus.sentences.add(sent)
                        sent2sent[sent] = sent
        print("sentences: %d" % len(corpus.sentences))
        self.tokenize(corpus)
        self.train_wv(corpus)

        for name, codes in code_dict.items():
            parser = self.parser_factory[name]
            print("{} starts.".format(parser.NAME))
            for path, code in tqdm(codes, desc=f' - ({parser.NAME})', leave=False, ascii=True):
                eles, rels = parser.parse(code, path)
                corpus.code_elements.update(eles)
                corpus.code_relations.update(rels)
        print("code elements: %d" % len(corpus.code_elements))
        print("code relations: %d" % len(corpus.code_relations))
        return corpus