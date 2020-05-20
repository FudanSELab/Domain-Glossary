#!/usr/bin/env python
# -*- coding: utf-8 -*-

class RelType:
    IS_A = "is a"
    HAS_A = "has a"
    RELATED_TO = "related to"


class Relation:
    def __init__(self, start, end, type):
        self.start = start
        self.end = end
        self.type = type

    def __str__(self):
        return f"<Relation: start={self.start}, end={self.end}, type={self.type}>"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)
