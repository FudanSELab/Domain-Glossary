#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from domainterm import App, Option
from domainterm.util.data_loader import DataLoader

import pickle
from pathlib import Path


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"

    # Configure general corpus
    Option.GENERAL_CORPUS = str(output_dir / "general.corpus")

    # Load htmls and codes
    htmls = [
        ("javadoc", "DL4J", "content"),
        ("html", "pytorch", "content"),
        ...
    ]
    codes = [
        ("java", "deeplearning4j/deeplearning4j-nn/.../LSTM.java", "content"),
        ("python", "torch/nn/../rnn.py", "content"),
        ...
    ]
    
    app = App()
    corpus = app.process(htmls, codes)

    concepts, relations = corpus.concepts, corpus.relations

    with Path(str(output_dir / "concepts.bin")).open("wb") as f:
        pickle.dump(concepts, f)

    with Path(str(output_dir / "relations.bin")).open("wb") as f:
        pickle.dump(relations, f)

    