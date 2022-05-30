import contextlib
import logging
import os
import random
import re
import string
from collections import defaultdict

import deepmatcher as dm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from certa.models.ermodel import ERModel


def wrapdm_mojito(model, ignore_columns=['label', 'id']):
    def wrapper(dataframe):
        data = dataframe.copy().drop([c for c in ignore_columns if c in dataframe.columns], axis=1)

        data['id'] = np.arange(len(dataframe))

        tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
        data.to_csv(tmp_name, index=False)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                data_processed = dm.data.process_unlabeled(tmp_name, trained_model=model,
                                                           ignore_columns=['ltable_id', 'rtable_id'])
                out_proba = model.run_prediction(data_processed, output_attributes=True)
                out_proba = out_proba['match_score'].values.reshape(-1)

        multi_proba = np.dstack((1 - out_proba, out_proba)).squeeze()

        os.remove(tmp_name)
        return multi_proba

    return wrapper


def wrapDm(test_df, model, given_columns=None, ignore_columns=['label', 'id', 'ltable_id', 'rtable_id'],
           outputAttributes=True, batch_size=4):
    if isinstance(test_df, csr_matrix):
        test_df = pd.DataFrame(data=np.zeros(test_df.shape))
        if given_columns is not None:
            test_df.columns = given_columns
    data = test_df.copy().drop([c for c in ignore_columns if c in test_df.columns], axis=1)
    names = []
    if data.columns[0] == 0:
        try:
            if given_columns is not None:
                data.columns = given_columns
            else:
                names = model.state_meta.all_left_fields + model.state_meta.all_right_fields
                data.columns = names
        except:
            pass

    if not ('id' in data.columns):
        data['id'] = np.arange(len(data))
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    data.to_csv(tmp_name, index=False)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            data_processed = dm.data.process_unlabeled(tmp_name, trained_model=model,
                                                       ignore_columns=['ltable_id', 'rtable_id', 'label', 'id',
                                                                       'originalRightId', 'alteredAttributes',
                                                                       'droppedValues', 'copiedValues'])
            predictions = model.run_prediction(data_processed, output_attributes=outputAttributes,
                                               batch_size=batch_size)
            out_proba = predictions['match_score'].values
    multi_proba = np.dstack((1 - out_proba, out_proba)).squeeze()
    os.remove(tmp_name)
    if outputAttributes:
        if len(names) == 0:
            names = list(test_df.columns)
        names.extend(['nomatch_score', 'match_score'])
        multi_proba_df = pd.DataFrame(multi_proba)
        if multi_proba_df.shape[0] != test_df.shape[0]:
            multi_proba_df = multi_proba_df.transpose()
        multi_proba_df.index = test_df.index
        full_df = pd.concat([test_df, multi_proba_df], axis=1, ignore_index=True, names=names)
        full_df.columns = names
        return full_df
    else:
        return multi_proba


def makeAttr(attribute, idx, isLeft):
    attr_prefixed = []
    for token in attribute.split():
        if isLeft:
            attr_prefixed.append('L' + str(idx) + '_' + token)
        else:
            attr_prefixed.append('R' + str(idx) + '_' + token)
    return " ".join(attr_prefixed)


def pairs_to_string(df, lprefix, rprefix, ignore_columns=['id', 'label']):
    pairs_string = []
    l_columns = [col for col in list(df) if (col.startswith(lprefix)) and (col not in ignore_columns)]
    r_columns = [col for col in list(df) if col.startswith(rprefix) and (col not in ignore_columns)]
    df = df.fillna("")
    for i in range(len(df)):
        this_row = df.iloc[i]
        this_row_str = []
        for j, lattr in enumerate(l_columns):
            this_attr = makeAttr(str(this_row[lattr]), j, isLeft=True)
            this_row_str.append(this_attr)
        for k, rattr in enumerate(r_columns):
            this_attr = makeAttr(str(this_row[rattr]), k, isLeft=False)
            this_row_str.append(this_attr)
        pairs_string.append(" ".join(this_row_str))
    return pairs_string


an_re = re.compile('[R|L]\d\_.+')


def makeRow(pair_str, attributes, lprefix, rprefix):
    row_map = defaultdict(list)
    for token in pair_str.split():
        if an_re.match(token):
            row_map[token[:2]].append(token[3:])
    row = {}
    for key in row_map.keys():
        if key.startswith('L'):
            ## key[1] is the index of attribute
            this_attr = lprefix + attributes[int(key[1])]
            row[this_attr] = " ".join(row_map[key])
        else:
            this_attr = rprefix + attributes[int(key[1])]
            row[this_attr] = " ".join(row_map[key])
    keys = dict.fromkeys(row.keys(), [])
    for r in keys:  # add any completely missing attribute (with '' value)
        if r.startswith(lprefix):
            twin_attr = 'r' + r[1:]
            if None == row.get(twin_attr):
                row[twin_attr] = ''
        elif r.startswith(rprefix):
            twin_attr = 'l' + r[1:]
            if None == row.get(twin_attr):
                row[twin_attr] = ''
    for a in attributes.values():
        try:
            if lprefix + a not in row:
                row[lprefix + a] = ''
            if rprefix + a not in row:
                row[rprefix + a] = ''
        except ValueError as e:
            pass
    return pd.Series(row)


def pairs_str_to_df(pairs_str_l, columns, lprefix, rprefix):
    lschema = list(filter(lambda x: x.startswith(lprefix), columns))
    schema = {}
    for i, s in enumerate(lschema):
        schema[i] = s.replace(lprefix, "")
    allTuples = []
    for pair_str in pairs_str_l:
        row = makeRow(pair_str, schema, 'ltable_', 'rtable_')
        allTuples.append(row)
    df = pd.DataFrame(allTuples)
    df['id'] = np.arange(len(df))
    return df


def pair_str_to_df(pair_str, columns, lprefix, rprefix):
    lschema = list(filter(lambda x: x.startswith(lprefix), columns))
    schema = {}
    for i, s in enumerate(lschema):
        schema[i] = s.replace(lprefix, "")
    row = makeRow(pair_str, schema, 'ltable_', 'rtable_')
    row['id'] = 0
    return pd.DataFrame(data=[row.values], columns=row.index)


an_re = re.compile('[R|L]\d\_.+')


def makeRow(pair_str, attributes, lprefix, rprefix):
    row_map = defaultdict(list)
    for token in pair_str.split():
        if an_re.match(token):
            row_map[token[:2]].append(token[3:])
    row = {}
    for key in row_map.keys():
        if key.startswith('L'):
            ## key[1] is the index of attribute
            this_attr = lprefix + attributes[int(key[1])]
            row[this_attr] = " ".join(row_map[key])
        else:
            this_attr = rprefix + attributes[int(key[1])]
            row[this_attr] = " ".join(row_map[key])
    keys = dict.fromkeys(row.keys(), [])
    for r in keys:  # add any completely missing attribute (with '' value)
        if r.startswith(lprefix):
            twin_attr = 'r' + r[1:]
            if None == row.get(twin_attr):
                row[twin_attr] = ''
        elif r.startswith(rprefix):
            twin_attr = 'l' + r[1:]
            if None == row.get(twin_attr):
                row[twin_attr] = ''
    for a in attributes.values():
        try:
            if lprefix + a not in row:
                row[lprefix + a] = ''
            if rprefix + a not in row:
                row[rprefix + a] = ''
        except ValueError as e:
            pass
    return pd.Series(row)


def pairs_str_to_df(pairs_str_l, columns, lprefix, rprefix):
    lschema = list(filter(lambda x: x.startswith(lprefix), columns))
    schema = {}
    for i, s in enumerate(lschema):
        schema[i] = s.replace(lprefix, "")
    allTuples = []
    for pair_str in pairs_str_l:
        row = makeRow(pair_str, schema, 'ltable_', 'rtable_')
        allTuples.append(row)
    df = pd.DataFrame(allTuples)
    df['id'] = np.arange(len(df))
    return df


def pair_str_to_df(pair_str, columns, lprefix, rprefix):
    lschema = list(filter(lambda x: x.startswith(lprefix), columns))
    schema = {}
    for i, s in enumerate(lschema):
        schema[i] = s.replace(lprefix, "")
    row = makeRow(pair_str, schema, 'ltable_', 'rtable_')
    row['id'] = 0
    return pd.DataFrame(data=[row.values], columns=row.index)


class DMERModel(ERModel):

    def __init__(self):
        super(DMERModel, self).__init__()
        self.name = 'dm'
        self.model = dm.MatchingModel(attr_summarizer='hybrid')

    def initialize_models(self, data):
        self.model.initialize(data)

    def train(self, label_train, label_valid, dataset_name):
        train_file = dataset_name + '_dm_train.csv'
        valid_file = dataset_name + '_dm_valid.csv'
        label_train.to_csv(train_file, index=False)
        label_valid.to_csv(valid_file, index=False)

        # read dataset
        trainLab, validationLab = dm.data.process(cache='models/saved/dm/' + dataset_name + '.pth', path='',
                                                  train=train_file,
                                                  validation=valid_file, left_prefix='ltable_',
                                                  right_prefix='rtable_')
        self.initialize_models(trainLab)

        logging.debug("TRAINING with {} samples", len(trainLab))
        # train default model with standard dataset
        self.model.run_train(trainLab, validationLab,
                             best_save_path='models/saved/dm/' + dataset_name + '_best_default_model.pth',
                             epochs=30)

        stats = self.model.run_eval(validationLab)
        os.remove(train_file)
        os.remove(valid_file)
        return stats

    def predict(self, x, mojito=False, expand_dim=False, **kwargs):
        if isinstance(x, np.ndarray):
            # data = to_deeper_data_np(x)
            x_index = np.arange(len(x))
            xc = pd.DataFrame(x, index=x_index)
        else:
            xc = x.copy()
        # if 'id' in xc.columns:
        #     xc = xc.drop(['id'], axis=1)
        res = wrapDm(xc, self.model, **kwargs)
        if mojito:
            res = np.dstack((res['nomatch_score'], res['match_score'])).squeeze()
            res_shape = res.shape
            if len(res_shape) == 1 and expand_dim:
                res = np.expand_dims(res, axis=1).T
        return res

    def evaluation(self, test_set):
        test_file = 'dm_test.csv'
        test_set.to_csv(test_file, index=False)

        # read dataset
        testLab = dm.data.process(path='', test=test_file, left_prefix='ltable_',
                                  right_prefix='rtable_', cache=None)

        f1 = self.model.run_eval(testLab)
        os.remove(test_file)
        return 0, 0, f1

    def load(self, path):
        if not path.endswith('.pth'):
            path = path + '.pth'
        self.model.load_state(path)

    def save(self, path):
        if not path.endswith('.pth'):
            path = path + '.pth'
        self.model.save_state(path, True)

    def predict_proba(self, x, **kwargs):
        return self.predict(x, mojito=True, expand_dim=True, **kwargs)
