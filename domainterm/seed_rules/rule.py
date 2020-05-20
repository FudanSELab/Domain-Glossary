#!/usr/bin/env python
# -*- conding: utf-8 -*-
from abc import abstractmethod, ABCMeta
import re
import textdistance


class Rule(metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.extract(*args)

    @abstractmethod
    def extract(self, sentence):
        raise NotImplementedError