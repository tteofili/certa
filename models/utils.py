import logging
import os
import pandas as pd

from certa.utils import merge_sources
from models.DeepER import DeepERModel
from models.bert import EMTERModel
from models.dm import DMERModel
from models.ermodel import ERModel


def from_type(type: str):
    model = ERModel()
    if "dm" == type:
        model = DMERModel()
    elif "deeper" == type:
        model = DeepERModel()
    elif "ditto" == type:
        model = EMTERModel()
    return model


def get_model(mtype: str, modeldir: str, datadir: str, modelname: str):
    model = from_type(mtype)

    os.makedirs(modeldir, exist_ok=True)

    print(f'working on {modelname}')
    print(f'reading data from {datadir}')

    lsource = pd.read_csv(datadir + '/tableA.csv')
    rsource = pd.read_csv(datadir + '/tableB.csv')
    gt = pd.read_csv(datadir + '/train.csv')
    valid = pd.read_csv(datadir + '/valid.csv')
    test = pd.read_csv(datadir + '/test.csv')

    print(f'data loaded')

    try:
        print(f'loading model from {modeldir}')
        ret_model = model.load(modeldir)
        if ret_model is None:
            print(f'no model found, now training')
            print('merging sources')
            train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
            test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])
            valid_df = merge_sources(valid, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
            print(f'training model with {len(train_df)} samples ({len(valid_df)} validation, {len(test_df)} test)')
            model.train(train_df, valid_df, modelname)
            print('evaluating model')
            precision, recall, fmeasure = model.evaluation(test_df)
            text_file = open(modeldir + 'report.txt', "a")
            text_file.write('p:' + str(precision) + ', r:' + str(recall) + ', f1:' + str(fmeasure))
            text_file.close()
            print('saving model')
            model.save(modeldir)
    except:
        pass

    return model
