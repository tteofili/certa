import re
from collections import defaultdict

import deepmatcher as dm
import numpy as np
import contextlib
import pandas as pd
import random
import os
import string


def wrapDm(test_df,model,ignore_columns=['label', 'id'],outputAttributes=True,batch_size=32):
    data = test_df.copy().drop([c for c in ignore_columns if c in test_df.columns],axis=1)
    if not('id' in data.columns):
        data['id'] = np.arange(len(data))
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    data.to_csv(tmp_name,index=False)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            data_processed = dm.data.process_unlabeled(tmp_name, trained_model = model,
                                                       ignore_columns=['ltable_id', 'rtable_id', 'label', 'id',
                                                                       'originalRightId', 'alteredAttributes'])
            predictions = model.run_prediction(data_processed, output_attributes= outputAttributes,\
                                              batch_size=batch_size)
            out_proba = predictions['match_score'].values
    multi_proba = np.dstack((1-out_proba, out_proba)).squeeze()
    os.remove(tmp_name)
    if outputAttributes:
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

class DMERModel():

    def __init__(self):
        #super(DMERModel, self).__init__()
        self.model = dm.MatchingModel(attr_summarizer='hybrid')

    def initialize_models(self, data):
        self.model.initialize(data)

    def classic_training(self, label_train, label_valid, dataset_name):
        train_file = dataset_name+'_dm_train.csv'
        valid_file = dataset_name+'_dm_valid.csv'
        label_train.to_csv(train_file, index=False)
        label_valid.to_csv(valid_file, index=False)

        # read dataset
        trainLab, validationLab = dm.data.process(cache=dataset_name + '.pth', path='', train=train_file,
                                                  validation=valid_file, left_prefix='ltable_',
                                                  right_prefix='rtable_')
        self.initialize_models(trainLab)

        print("TRAINING with " + str(len(trainLab)) + " samples")
        # train default model with standard dataset
        self.model.run_train(trainLab, validationLab, best_save_path=dataset_name + '_best_default_model.pth')

        stats = self.model.run_eval(validationLab)

        return stats

    def predict(self, x, **kwargs):
        xc = x.copy()
        # if 'id' in xc.columns:
        #     xc = xc.drop(['id'], axis=1)
        return wrapDm(xc, self.model, **kwargs)

    def load(self, path):
        self.model.load_state(path)

    def save(self, path):
        self.model.save_state(path, True)
