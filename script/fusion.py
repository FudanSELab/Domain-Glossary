#!/usr/bin/env python
# -*- coding: utf-8 -*-

from domainterm.pipeline import AliasFusion

import pickle
from pathlib import Path


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "output"

    with Path(str(output_dir / "term_recognizer.corpus")).open("rb") as f:
        corpus = pickle.load(f)

    fusion = AliasFusion()
    corpus = fusion.process(corpus)

    with Path(str(output_dir / "alias_fusion.corpus")).open("wb") as f:
        pickle.dump(corpus, f)

    