#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import itertools
from pathlib import Path
import ast

from .base_parser import BaseParser


class PythonParser(BaseParser):
    NAME = "PythonParser"

    def __init__(self):
        super(PythonParser, self).__init__()

        self.modules = set()

    def parse(self, code, file_path=None):
        elements = set()
        relations = set()
        if file_path.endswith("__init__.py"):
            return elements, relations

        module_eles = [self.split_camel(str(part)) for part in Path(re.sub(r"\.py", "", file_path)).parts]
        if len(module_eles) == 0:
            return elements, relations
        elements.update(module_eles)

        relations.update([(head, tail, "has a") for head, tail in itertools.combinations(module_eles, 2)])

        try:
            tree = ast.parse(code)
        except Exception:
            # print(code)
            return elements, relations
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                for alias in node.names:
                    elements.add(self.split_camel(alias.name))
                    relations.add((module_eles[-1], self.split_camel(alias.name), "has a"))
                    if alias.asname:
                        elements.add(self.split_camel(alias.asname))
                        relations.add((module_eles[-1], self.split_camel(alias.asname), "has a"))
                        relations.add((self.split_camel(alias.asname), self.split_camel(alias.name), "is a"))
            elif isinstance(node, ast.ClassDef):
                elements.add(self.split_camel(node.name))
                relations.update([(ele, self.split_camel(node.name), "has a") for ele in module_eles])
                for base in node.bases:
                    if isinstance(base, ast.Attribute):
                        relations.add((self.split_camel(node.name), self.split_camel(base.attr), "is a"))
                    elif isinstance(base, ast.Name):
                        relations.add((self.split_camel(node.name), self.split_camel(base.id), "is a"))
        return elements, relations