import os
import traceback

import pandas as pd

from certa.utils import merge_sources
from metrics.faithfulness import get_faithfullness
from models.utils import from_type

model_type = 'emt'
experiments_dir = 'quantitative/'
root_datadir = 'datasets/'
base_dir = ''
samples = 50
whitelist = ['abt_buy']

for subdir, dirs, files in os.walk(experiments_dir):
    for dataset in dirs:
        if dataset not in whitelist:
            continue
        datadir = os.path.join(root_datadir, dataset)
        test = pd.read_csv(datadir + '/test.csv')

        lsource = pd.read_csv(datadir + '/tableA.csv')
        rsource = pd.read_csv(datadir + '/tableB.csv')
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])[:samples]

        model = from_type('%s' % model_type)
        try:
            model.load('%smodels/%s/%s' % (base_dir, model_type, dataset))

            faithfulness = get_faithfullness(model, '%s%s%s/%s' % (base_dir, experiments_dir, dataset, model_type),
                                             test_df)
            print(f'{model_type}: faithfulness for {dataset}: {faithfulness}')
        except:
            print(traceback.format_exc())
            print(f'skipped {dataset}')
            pass
