#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def html(dirnames):
        dirnames = [name for name in dirnames if len(name) > 0]
        htmls = []
        for dirname in dirnames:
            for subdir in Path(dirname).glob("*"):
                print("Loading %s..." % str(subdir))
                doc_type = "javadoc" if "javadoc" in subdir.parts[-1] else "html"
                for fn in Path(subdir).glob("*"):
                    try:
                        with Path(fn).open("r", encoding="utf-8") as f:
                            html = f.read().strip()
                        index = str(fn).find(dirname)
                        short_fn = str(fn)[index:] if index > -1 else str(fn)
                        htmls.append((doc_type, short_fn, html))
                    except Exception:
                        pass
        return htmls

    @staticmethod
    def code(dirnames):
        dirnames = [name for name in dirnames if len(name) > 0]
        codes = []
        for dirname in dirnames:
            print("Loading %s..." % str(dirname))
            for fn in Path(dirname).rglob("*.*"):
                if fn.is_file():
                    try:
                        with Path(fn).open("r", encoding="utf-8") as f:
                            code = f.read().strip()
                        index = str(fn).find(dirname)
                        if index > -1:
                            index += len(dirname)
                        short_fn = str(fn)[index:] if index > -1 else str(fn)
                        short_fn = short_fn.lstrip("/\\")
                        if fn.suffix == ".py":
                            codes.append(("python", short_fn, code))
                        elif fn.suffix == ".java":
                            codes.append(("java", short_fn, code))
                    except Exception:
                        continue
        return codes