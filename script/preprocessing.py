#!/usr/bin/env python
# -*- coding: utf-8 -*-

from domainterm.pipeline import Preprocessor
from domainterm.util.data_loader import DataLoader

import pickle
from pathlib import Path


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"

    htmls = DataLoader.html([str(data_dir / "dl4j/html-data"), str(data_dir / "pytorch/html-data"), str(data_dir / "tf/html-data")])
    codes = DataLoader.code([str(data_dir / "dl4j/source-code"), str(data_dir / "pytorch/source-code"), str(data_dir / "tf/source-code")])

    # htmls = htmls[:10]
    # codes = codes[:10]
    
    preprocessor = Preprocessor()
    corpus = preprocessor.process(htmls, codes)
    with Path(str(output_dir / "preprocessor.corpus")).open("wb") as f:
        pickle.dump(corpus, f)