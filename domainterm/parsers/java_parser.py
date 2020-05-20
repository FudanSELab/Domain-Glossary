#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import itertools
from pathlib import Path

import javalang
from javalang.tree import *

from .base_parser import BaseParser
from ..types import RelType

class JavaParser(BaseParser):
    NAME = "JavaParser"

    def __init__(self, **cfg):
        super(JavaParser, self).__init__()

    def parse(self, code, file_path=None):
        elements = set()
        relations = set()

        try:
            tree = javalang.parse.parse(code)
        except Exception:
            # print(code)
            return elements, relations

        package_eles = None
        for _, node in tree:
            if isinstance(node, PackageDeclaration):
                package = node.name
                package_eles = [self.split_camel(ele) for ele in package.split(".")]
                elements.update(package_eles)
                relations.update([(head, tail, "has a") for head, tail in itertools.combinations(package_eles, 2)])
                break
        else:
            return elements, relations

        main_class = str(Path(re.sub(r"\.java", "", file_path)).parts[-1])
        elements.add(self.split_camel(main_class))
        for _, node in tree:
            if isinstance(node, Import):
                imported_class = node.path.split(".")[-1]
                relations.add((self.split_camel(main_class), self.split_camel(imported_class), "has a"))

        processed = set()
        def __parse_class(clazz, class_name):
            # processed.add(clazz)
            for _, node in clazz:
                if node in processed:
                    continue
                processed.add(node)
                if isinstance(node, ClassDeclaration) and node != clazz:
                    elements.add(self.split_camel(node.name))
                    relations.add((self.split_camel(class_name), self.split_camel(node.name), "has a"))
                    __parse_class(node, node.name)
                elif isinstance(node, FieldDeclaration):
                    relations.add((self.split_camel(class_name), self.split_camel(node.type.name), "has a"))

        for clazz in tree.types:
            if isinstance(clazz, ClassDeclaration):
                elements.add(self.split_camel(clazz.name))
                relations.update([(p_ele, self.split_camel(clazz.name), "has a") for p_ele in package_eles])
                if clazz.extends:
                    relations.add((self.split_camel(clazz.name), self.split_camel(clazz.extends.name), "is a"))
                __parse_class(clazz, clazz.name)
        return elements, relations


