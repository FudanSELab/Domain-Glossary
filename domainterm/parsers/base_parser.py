#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import re
import functools


class BaseParser(metaclass=ABCMeta):
    NAME = "Parser"
    
    def __init__(self):
        self.cache = {}

    def __call__(self, *args):
        return self.parse(*args)

    @abstractmethod
    def parse(self, code, file_path=None):
        pass

    @functools.lru_cache(maxsize=10000)
    def split_camel(self, camel_case:str, to_lower=True):
        space_delimited = re.sub(r'_', " ", camel_case).strip()
        space_delimited = re.sub(r'([A-Za-z])([Vv][0-9]+)([A-Za-z]|$)', r'\1 \2 \3', space_delimited)
        space_delimited = re.sub(r'([A-Za-z])([0-9]+D)([A-Z]|$)', r'\1 \2 \3', space_delimited)
        space_delimited = re.sub(r'([A-Z][0-9]?)(to)([A-Z]|$)', r'\1 \2 \3', space_delimited)
        space_delimited = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', space_delimited)
        space_delimited = re.sub(r'([A-Z]+)', r' \1', space_delimited)
        space_delimited = re.sub(r'([A-UW-Za-uw-z])(2)([A-Za-z]|\s)', r'\1 To \3', space_delimited)
        space_delimited = re.sub(r'\s+', ' ', space_delimited)
        space_delimited = re.sub(r'([A-UW-Za-uw-z])(4)([A-Za-z]|\s)', r'\1 For \3', space_delimited)
        space_delimited = re.sub(r'\s+', ' ', space_delimited)
        space_delimited = re.sub(r'([A-Za-z]) ([Vv][0-9]+)', r'\1\2', space_delimited)
        space_delimited = re.sub(r'(\s|^|[A-Z])([0-9]+) ([A-Z])', r'\1\2\3', space_delimited)
        if to_lower:
            space_delimited = space_delimited.lower()
        return space_delimited