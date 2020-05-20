#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GloVe Embeddings + chars conv and max pooling + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
from pathlib import Path

import numpy as np
import tensorflow as tf
# from concept_recognizer.crf import crf_decode
# from tf_metrics import precision, recall, f1

from .masked_conv import masked_conv1d_and_max


def parse_fn(words, tags):
    # Encode in Bytes for TF
    # print(words, tags)
    _words = [word.encode() for word in words]
    tags = [tag.encode() for tag in tags]

    # Chars
    chars = [[c.encode() for c in word] for word in words]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((_words, len(_words)), (chars, lengths)), tags


def generator_fn(pairs):
    for words, tags in pairs:
        yield parse_fn(words, tags)


def input_fn(pairs, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, pairs),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # Read vocabs and inputs
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params['words']), num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params['chars']), num_oov_buckets=params['num_oov_buckets'])
    indices = [idx for idx, tag in enumerate(params['tags']) if tag.strip() != 'O']
    num_tags = len(indices) + 1
    num_chars = sum(1 for _ in params['chars']) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                        training=training)

    # Char LSTM
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    # emb = np.load(params['emb'])['embeddings']  # np.array
    variable = np.vstack([params['emb'], [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, scores = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant(params['tags']))
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        # transition_params = vs.get_variable("transitions", [num_tags, num_tags])
        # probs = tf.contrib.crf.crf_sequence_score(logits, pred_ids, nwords, transition_params)
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings,
            'score': scores
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_tensor(tf.constant(params['tags']))
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': tf.metrics.precision(tags, pred_ids, weights),
            'recall': tf.metrics.recall(tags, pred_ids, weights),
            # 'loss': loss
            # 'f1': tf.contrib.metrics.f1_score(tags, pred_ids, weights),
        }
        # for metric_name, op in metrics.items():
        #     tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)
