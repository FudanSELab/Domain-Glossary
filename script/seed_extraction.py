#!/usr/bin/env python
# -*- coding: utf-8 -*-

from domainterm import Option
from domainterm.pipeline import SeedExtractor

import pickle
from pathlib import Path

if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "output"

    with Path(str(output_dir / "preprocessor.corpus")).open("rb") as f:
        corpus = pickle.load(f)

    Option.GENERAL_CORPUS = "output/general.corpus"
    extractor = SeedExtractor()
    corpus = extractor.process(corpus)

    with Path(str(output_dir / "seed_extractor.corpus")).open("wb") as f:
        pickle.dump(corpus, f)