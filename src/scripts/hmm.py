from __future__ import division

import os
import string
import sys

import settings
import viterbi

from collections import defaultdict

# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

# Additive smoothing parameter
alpha = 0.001


def generate_vocab(min_cnt=2, train_fp=settings.TRAIN):
    """
    Generate vocabulary
    """
    vocab = defaultdict(int)

    with open(train_fp, "r") as train:

        for line in train:

            # Ignore empty lines
            if not line.split():
                continue
            tok, tag = line.split("\t")
            vocab[tok] += 1

    # Get list of vobulary words
    vocab = [k for k, v in vocab.items() if v >= min_cnt]

    # Add newline/unknown word tokens
    vocab.extend([line.strip() for line in open(settings.UNK_TOKS, "r")])

    # Sort
    vocab = sorted(vocab)

    with open(settings.VOCAB, "w") as out:
        for word in vocab:
            out.write("{0}\n".format(word))
    out.close()

    return vocab


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"


def train_model(vocab, train_fp=settings.TRAIN):
    """
    Train part-of-speech (POS) tagger model
    """
    vocab = set(vocab)
    emiss = defaultdict(int)
    trans = defaultdict(int)
    context = defaultdict(int)

    with open(train_fp, "r") as train:

        # Start state
        prev = "--s--"

        for line in train:

            # End of sentence
            if not line.split():
                word = "--n--"
                tag = "--s--"

            else:
                word, tag = line.split()
                # Handle unknown words
                if word not in vocab:
                    word = assign_unk(word)

            trans[" ".join([prev, tag])] += 1
            emiss[" ".join([tag, word])] += 1
            context[tag] += 1
            prev = tag

    model = []
