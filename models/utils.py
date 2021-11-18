import logging

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

    print(f'working on {modelname}')
    logging.info(f'reading data from {datadir}')

    lsource = pd.read_csv(datadir + '/tableA.csv')
    rsource = pd.read_csv(datadir + '/tableB.csv')
    gt = pd.read_csv(datadir + '/train.csv')
    valid = pd.read_csv(datadir + '/valid.csv')
    test = pd.read_csv(datadir + '/test.csv')

    try:
        logging.info('loading model from {}', modeldir)
        model.load(modeldir)
    except:
        logging.info('training model')
        train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])
        valid_df = merge_sources(valid, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
        model.train(train_df, valid_df, modelname)

        precision, recall, fmeasure = model.evaluation(test_df)
        text_file = open(modeldir + 'report.txt', "a")
        text_file.write('p:' + str(precision) + ', r:' + str(recall) + ', f1:' + str(fmeasure))
        text_file.close()
        model.save(modeldir)
    return model
