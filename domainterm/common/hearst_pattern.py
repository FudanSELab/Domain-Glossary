#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from .spacy_nlp import SpacyNLP


class HearstPatterns:
    def __init__(self, extended=False):
        self.__nlp = SpacyNLP.get_inst()
        self.__adj_stopwords = set([
            'able', 'available', 'brief', 'certain', 'different', 'due', 'enough', 'especially',
            'few', 'fifth', 'former', 'his', 'howbeit', 'immediate', 'important', 'inc', 'its',
            'last', 'latter', 'least', 'less', 'likely', 'little', 'many', 'ml', 'more', 'most',
            'much', 'my', 'necessary', 'new', 'next', 'non', 'old', 'other', 'our', 'ours', 'own',
            'particular', 'past', 'possible', 'present', 'proud', 'recent', 'same', 'several',
            'significant', 'similar', 'such', 'sup', 'sure'
        ])

        # now define the Hearst patterns
        # format is <hearst-pattern>, <general-term>
        # so, what this means is that if you apply the first pattern, the firsr Noun Phrase (NP)
        # is the general one, and the rest are specific NPs
        self.__is_a_patterns = [
            (r'(NP_\w+(,|and|or)? )+(and|or) other NP_\w+', 'last'),
            (r'NP_\w+ (is|was) a NP_\w+', 'last'),
            (r'NP_\w+,? including (NP_\w+(,|and|or)? ?)+', 'first'),
            (r'NP_\w+,? especially (NP_\w+(,|and|or)? ?)+', 'first'),
            (r'NP_\w+,? such as (NP_\w+(,|and|or)? ?)+', 'first'),
            (r'such NP_\w+ as (NP_\w+(,|and|or)? ?)+', 'first'),
        ]

        if extended:
            self.__hearst_patterns.extend([
                (r'((NP_\w+ ?(, )?)+(and |or )?any other NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?some other NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?be a NP_\w+)', 'last'),
                (r'(NP_\w+ (, )?like (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'such (NP_\w+ (, )?as (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )?like other NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?one of the NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?one of these NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?one of those NP_\w+)', 'last'),
                (r'example of (NP_\w+ (, )?be (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )?be example of NP_\w+)', 'last'),
                (r'(NP_\w+ (, )?for example (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )?wich be call NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?which be name NP_\w+)', 'last'),
                (r'(NP_\w+ (, )?mainly (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?mostly (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?notably (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?particularly (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?principally (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?in particular (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?except (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?other than (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?e.g. (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?i.e. (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )?a kind of NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?kind of NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?form of NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?which look like NP_\w+)', 'last'),
                (r'((NP_\w+ ?(, )?)+(and |or )?which sound like NP_\w+)', 'last'),
                (r'(NP_\w+ (, )?which be similar to (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?example of this be (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?type (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )? NP_\w+ type)', 'last'),
                (r'(NP_\w+ (, )?whether (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(compare (NP_\w+ ?(, )?)+(and |or )?with NP_\w+)', 'last'),
                (r'(NP_\w+ (, )?compare to (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'(NP_\w+ (, )?among -PRON- (NP_\w+ ? (, )?(and |or )?)+)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )?as NP_\w+)', 'last'),
                (r'(NP_\w+ (, )? (NP_\w+ ? (, )?(and |or )?)+ for instance)', 'first'),
                (r'((NP_\w+ ?(, )?)+(and |or )?sort of NP_\w+)', 'last')
            ])

    def chunk(self, sentence):
        doc = self.__nlp.parse(sentence)
        sentence_text = " " + " ".join([t.lemma_ for t in doc])
        for chunk in doc.noun_chunks:
            chunk_arr = [token.lemma_ for token in chunk]
            chunk_text = " ".join(chunk_arr).strip()
            if len(chunk_text) == 0:
                continue
            # print("chunk_lemma:", chunk_lemma)
            replacement_value = "NP_" + "_".join(chunk_arr)
            sentence_text = sentence_text.replace(" {} ".format(chunk_text), " {} ".format(replacement_value))
        return sentence_text.strip()

    """
        This is the main entry point for this code.
        It takes as input the rawtext to process and returns a list of tuples (specific-term, general-term)
        where each tuple represents a hypernym pair.
    """

    def find_hyponyms(self, sentence):

        hyponyms = []
        np_tagged_sentence = self.chunk(sentence)

        for (hearst_pattern, parser) in self.__is_a_patterns:
            # print(hearst_pattern)
            matches = re.search(hearst_pattern, np_tagged_sentence)
            if matches:
                match_str = matches.group(0)
                # print(match_str)

                nps = [a for a in match_str.split() if a.startswith("NP_")]

                if parser == "first":
                    general = nps[0]
                    specifics = nps[1:]
                else:
                    general = nps[-1]
                    specifics = nps[:-1]
                    # print(str(general))
                    # print(str(nps))
                general = self.clean_hyponym_term(general)
                for i in range(len(specifics)):
                    #print("%s, %s" % (specifics[i], general))
                    specific = self.clean_hyponym_term(specifics[i])
                    if specific is None:
                        continue
                    hyponyms.append((specific, general))
        return hyponyms

    def clean_hyponym_term(self, term):
        # good point to do the stemming or lemmatization
        term = term.replace("NP_", "").replace("_", " ")
        chunk_arr = term.split()
        while len(chunk_arr) > 0:
            if chunk_arr[0] in self.__adj_stopwords:
                chunk_arr.pop(0)
            else:
                break
        while len(chunk_arr) > 0:
            if chunk_arr[-1] in self.__adj_stopwords:
                chunk_arr.pop(-1)
            else:
                break
        if len(chunk_arr) > 0:
            return " ".join(chunk_arr)
        return None
