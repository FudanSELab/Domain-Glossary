#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta


class Pipe(metaclass=ABCMeta):
    def __init__(self, name="pipe", **cfg):
        self.name = name
    
    def __call__(self, *args, **kwagrs):
        self.process(*args, **kwagrs)

    def with_name(self, name):
        self.name = name
        return self

    @abstractmethod
    def process(self, *args, **kwagrs):
        pass
