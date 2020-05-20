#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from domainterm import App, Option
from domainterm.util.data_loader import DataLoader

import pickle
from pathlib import Path


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"

    Option.GENERAL_CORPUS = "output/general.corpus"

    htmls = DataLoader.html([str(data_dir / "dl4j/html-data"), str(data_dir / "pytorch/html-data"), str(data_dir / "tf/html-data")])
    codes = DataLoader.code([str(data_dir / "dl4j/source-code"), str(data_dir / "pytorch/source-code"), str(data_dir / "tf/source-code")])

    app = App()
    corpus = app.process(htmls, codes)

    concepts, relations = corpus.concepts, corpus.relations

    with Path(str(output_dir / "concepts.bin")).open("wb") as f:
        pickle.dump(concepts, f)

    with Path(str(output_dir / "relations.bin")).open("wb") as f:
        pickle.dump(relations, f)

    