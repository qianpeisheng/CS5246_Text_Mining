# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on NQ."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import json_lines
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class NQExample(object):
    """
    A single training/test example for the natural questions dataset.
    For examples without an answer, the start and end position are -1.

    Modified from SquadExample
    NQ: no qas_id;
    For now, only load short answers;
    Use only tokens instead of chars
    the end token is not in the final answer, but for the convenience of list indexing
    
    Also to limit the difficulty of this project
    i.e. list[a,b] is right exclusive

    (c,s,e,t)
    t is question type
    0 = short; 1 = long; 2 = yes; 3 = no; 4 = no-answer

    in the run_SQUAD.py, the word 'token' sometime times refers to the annotated answer
    
    for evaluation NQ need example id
    """

    def __init__(self,
                 question_tokens,
                 doc_tokens,
                 orig_answer_text=None,
                 start_token=None,
                 end_token=None,
                 question_type = 0,
                 example_id = -1):
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_token = start_token
        self.end_token = end_token
        self.question_type = question_type
        self.example_id = example_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += ", question_text: %s" % (
            self.question_tokens)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_token:
            s += ", start_token: %d" % (self.start_token)
        if self.end_token:# should this be end token ?
            s += ", end_token: %d" % (self.end_token)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_token=None,
                 end_token=None,
                ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_token = start_token
        self.end_token = end_token



def read_NQ_examples(input_file, is_training):
    """Read a NQ json line file into a list of NQExamples
."""

    def add_special_markup_tokens(example):
        """ add '[Paragraph=N]', '[Table=N]', and '[List=N]' at the beginning of the N-th paragraph, list and
        table respectively inthe document. For details, refer to the paper.
        Returns the doc tokens 
        """
        
        p_id = 0
        t_id = 0
        l_id = 0

        html_table_list = ['<Table>', '<table>']
        html_paragraph_list = ['<P>', '<p>']
        html_list_list = ['<Ul>', '<Ol>', '<ul>', '<ol>']
        orig_doc_tokens = example['document_tokens']

        # construct new doc_token
        doc_tokens = []
        for orig_doc_token in orig_doc_tokens:
            token = orig_doc_token['token']
            if token in html_table_list:
                doc_tokens.append('[Table={}]'.format(t_id))
                t_id += 1

            elif token in html_paragraph_list:
                doc_tokens.append('[Paragraph={}]'.format(p_id))
                p_id += 1

            elif token in html_list_list:
                doc_tokens.append('[List={}]'.format(l_id))
                l_id += 1

            
            # unify html tags, this needs to be discussed
            if orig_doc_token['html_token']:
                token = '[html_token]'
            doc_tokens.append(token)
        
        return doc_tokens
    
    def get_offset(tokens, index):
        """
        Get offset of answer spans, after special markup tokens are added
        """

        p_id = 0
        t_id = 0
        l_id = 0

        html_table_list = ['<Table>', '<table>']
        html_paragraph_list = ['<P>', '<p>']
        html_list_list = ['<Ul>', '<Ol>', '<ul>', '<ol>']
        for token in tokens[0:index]:
            if token['token'] in html_table_list:
                doc_tokens.append('[Table={}]'.format(t_id))
                t_id += 1

            elif token['token'] in html_paragraph_list:
                doc_tokens.append('[Paragraph={}]'.format(p_id))
                p_id += 1

            elif token['token'] in html_list_list:
                doc_tokens.append('[List={}]'.format(l_id))
                l_id += 1

        offset = t_id + p_id + l_id
        return offset

    with open(input_file, "r", encoding='utf-8') as f:
        examples = []
        count = 0
        for item in json_lines.reader(f):

            # load relevant fields
            question_tokens = item['question_tokens']
            # question token are normal tokens
            # doc tokens are:
            # form {'end_byte': 95, 'html_token': False, 'start_byte': 92, 'token': 'The'}
            # simplify doc_tokens
            doc_tokens = add_special_markup_tokens(item)
            old_doc_tokens = item['document_tokens']

            short_answers = item['annotations'][0]['short_answers']
            long_answer = item['annotations'][0]['long_answer']
            example_id = item['example_id']
            
            # decide question type
            question_type = 4 # assume it is a no-answer question
            if item['annotations'][0]['long_answer']['start_token'] != -1:
                question_type = 1 # long answer
            if len(short_answers) > 0:
                question_type = 0 # short answer
            if len(item['annotations'][0]['yes_no_answer']) == 3:
                question_type = 2 # yes
            if len(item['annotations'][0]['yes_no_answer']) == 2:
                question_type = 3 # no
            
            # split short answers and build examples
            if len(short_answers) > 0:
                for short_answer in short_answers:
                    start_token = short_answer['start_token']
                    end_token = short_answer['end_token']
                    # print(start_token, end_token, 'start, end orig')
                    # i_str = ''
                    # for i in old_doc_tokens[start_token: end_token]:
                    #     i_str += i['token']
                    # print(i_str, ' orig answer')
                    start_token += get_offset(old_doc_tokens, start_token)
                    end_token += get_offset(old_doc_tokens, end_token)
                    # print(start_token, end_token, 'start end with offset')
                    # i_str = ''
                    # for i in doc_tokens[start_token: end_token]:
                    #     i_str += i
                    # print(i_str, 'with offset, answer')
                    # print('............................................')
                    orig_answer_text = ''
                    for i in range(start_token, end_token):
                        orig_answer_text += doc_tokens[i]
                        orig_answer_text += ' ' # Not correct if there is no space, e.g. 'Hi!', for debugging only
                    example = NQExample(question_tokens=question_tokens,
                            doc_tokens=doc_tokens,
                            orig_answer_text=orig_answer_text,
                            start_token=start_token,
                            end_token=end_token,
                            question_type=question_type,
                            example_id=example_id)
                    examples.append(example)
            elif long_answer['candidate_index'] != -1:
                # build long answer examples
                start_token = long_answer['start_token']
                end_token = long_answer['end_token']
                # print(start_token, end_token, 'start, end orig')
                # i_str = ''
                # for i in old_doc_tokens[start_token: end_token]:
                #     i_str += i['token']
                # print(i_str, ' orig answer')
                start_token += get_offset(old_doc_tokens, start_token)
                end_token += get_offset(old_doc_tokens, end_token)
                # i_str = ''
                # for i in doc_tokens[start_token: end_token]:
                #         i_str += i
                #print(i_str, 'with offset, answer')
                orig_answer_text = ''
                for i in range(start_token, end_token):
                    orig_answer_text += doc_tokens[i]
                    orig_answer_text += ' ' # Not correct if there is no space, e.g. 'Hi!', for debugging only
                example = NQExample(question_tokens=question_tokens,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_token=start_token,
                        end_token=end_token,
                        question_type=question_type)
                examples.append(example)

            count += 1
            # if len(examples) > 2:
            #     break
            if count % 500 == 0:
                print('number of examples loaded: {}'.format(count))
            
        print('Total number of examples loaded: {}'.format(len(examples)))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s.
    The main difference between NQ and SQUAD is that
    NQ is tokenized, so the process to tokenize (using bytes) is not necessary
    though bytes are also provided

    When evaluating, no need to augment
    """

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = example.question_tokens

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            # subwords are compulsory
            # not for html tags ?
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens): # should it be loosen to >= ?
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_token = None
            end_token = None

            # if not is_training: # is evaluating
            #     out_of_span = False # the annotation must be within the text
            #     doc_start = doc_span.start
            #     doc_end = doc_span.start + doc_span.length -1


            if is_training or True: # use this for evaluation for now
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.

                # The above is for SQUAD
                # for NQ, downsample by 50 times
                # for now , throw them out; check the paper for details
                # what is the right inclusion/ exclusion setting in SQUAD?
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False

                # beware that the new start token is the start of its sub token
                # not the original token
                sub_start_token = orig_to_tok_index[example.start_token]
                sub_end_token = None
                if example.end_token < len(orig_to_tok_index) - 1:
                    sub_end_token = orig_to_tok_index[example.end_token + 1] - 1
                else:
                    sub_end_token = len(all_doc_tokens) - 1
                if not (sub_start_token >= doc_start and
                        sub_end_token <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_token = 0
                    end_token = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_token = sub_start_token - doc_start + doc_offset
                    end_token = sub_end_token - doc_start + doc_offset
            # change the logging scheme to track preprocessing progress 
            
            # the baseline downsamples no-answer instances by 50 times

            add_feature = False
            if not out_of_span:
                add_feature = True
            elif is_training: # do not add during evaluation
                flag = random.randint(1,51)
                if flag == 23: # or any number in (1, 51)
                    add_feature = True
            else: # eval
                add_feature = True
                    
            if add_feature:
                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_token=start_token,
                        end_token=end_token,
                        ))
                unique_id += 1
                if start_token != 0: # log only those with answers
                    logger.info("\n*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info('doc start, doc end')
                    logger.info(doc_start)
                    logger.info(doc_end)
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    #logger.info("tokens: %s" % " ".join(tokens))
                    #logger.info("token_to_orig_map: %s" % " ".join([
                    #    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    #logger.info("token_is_max_context: %s" % " ".join([
                    #    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    #]))
                    #logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    #logger.info(
                    #    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    #logger.info(
                    #    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training:
                        answer_text = " ".join(tokens[start_token:end_token]) # SQUAD uses end + 1; NQ seems not
                        logger.info("start_token: %d" % (start_token))
                        logger.info("end_token: %d" % (end_token))
                        logger.info(
                            "answer: %s" % (answer_text))

    return features

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    nq_json = {'predictions':[]}


    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # logger.info(str(start_indexes[0]))
            # logger.info(str(end_indexes[0]))

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text",'start_index', 'end_index', "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            # logger.info(pred)
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_index=pred.start_index,
                    end_index=pred.end_index,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
            # logger.info(nbest)
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(1, # change 0 to 1, i.e. append instead of prepend, otherwise empty will be saved
                    _NbestPrediction(text="empty", start_index = 0, end_index = 0, start_logit=0.0, end_logit=0.0))
                
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_index = 0, end_index = 0, start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            #output['example_id'] = example.example_id
            output["text"] = entry.text
            output["probability"] = probs[i]
            output['start_index'] = entry.start_index
            output['end_index'] = entry.end_index
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)
        
        for (i, entry) in enumerate(nbest):
            # for NQ, need only nbest 0
            nq_output = collections.OrderedDict()
            nq_output['example_id'] = example.example_id
            #nq_output['long_answer'] = {'start_byte:':0, 'end_byte'}
            nq_output['short_answers'] = []
            nq_output["short_answers_score"] = probs[i]
            # we do not use byte, thus set to -1, see the test code comment
            start_index = entry.start_index
            end_index = entry.end_index
            if start_index == end_index:
                start_index -= 1
                end_index += 1
            # TODO the evaluation script does not allow start = end?
            nq_output['short_answers'].append({'start_byte':-1,
                                            'end_byte':-1,
                                            'start_token':start_index,
                                            'end_token':end_index})
            # currently do not use long answer
            # but can be added
            nq_output['long_answer'] = {'start_byte':-1,
                                        'end_byte':-1,
                                        'start_token':-1,
                                        'end_token':-1}
            nq_output["long_answer_score"] = 1 # add this field to avoid error
                        
            nq_output['yes_no_score'] = 'NONE'
            if nq_output['example_id'] != -1:
                nq_json['predictions'].append(nq_output)
            
            break # take only the best answer
        # TODO why some has id -1?
        # for od in nbest_json:
        #     # logger.info(str(od['probability']))
        #     logger.info(od['text'])
        #     logger.info(od['probability'])

        assert len(nbest_json) >= 1

        # replace qas_id in SQUAD with example index
        all_predictions[example_index] = nbest_json[0]["text"]
        #all_predictions[example_index] = nq_json[0]


    with open(output_prediction_file, "w") as writer:
        # writer.write(json.dumps(all_predictions, indent=4) + "\n")
        writer.write(json.dumps(nq_json, indent=4) + "\n")
        # TODO html tokens and some helper tokens are not removed

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    # SQUAD use 384, NQ uses 512
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    #import pdb; pdb.set_trace();
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_NQ_examples(
            input_file=args.train_file, is_training=True)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForQuestionAnswering.from_pretrained(args.bert_model,
                cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank)))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_tokens = torch.tensor([f.start_token for f in train_features], dtype=torch.long)
        #import pdb; pdb.set_trace()
        all_end_tokens = torch.tensor([f.end_token for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_tokens, all_end_tokens)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForQuestionAnswering(config)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForQuestionAnswering.from_pretrained(args.bert_model)

    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_NQ_examples(
            input_file=args.predict_file, is_training=False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                # logger.info(str(batch_start_logits))
                # logger.info(str(batch_end_logits))
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.null_score_diff_threshold)


if __name__ == "__main__":
    main()
