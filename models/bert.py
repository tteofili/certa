import logging
import os
import string
import random
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor

import emt.model
import emt.config
import emt.data_representation
import emt.data_loader
import emt.optimizer
import emt.evaluation
import emt.torch_initializer
import emt.training
import emt.prediction
from models.ermodel import ERModel

BATCH_SIZE = 8

MAX_SEQ_LENGTH = 250


def emt_mojito_predict(model):
    def wrapper(dataframe):
        dataframe['id'] = np.arange(len(dataframe))
        output = model.predict(dataframe)
        return np.dstack((output['nomatch_score'], output['match_score'])).squeeze()

    return wrapper


class EMTERModel(ERModel):

    def __init__(self):
        self.name = 'emt'
        super(EMTERModel, self).__init__()
        self.model_type = 'distilbert'
        config_class, model_class, tokenizer_class = emt.config.Config().MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained('distilbert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.model = model_class.from_pretrained('distilbert-base-uncased', config=config)

    def train(self, label_train, label_valid, dataset_name):
        device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)

        self.model = self.model.to(device)

        exp_dir = 'models/emt/' + dataset_name
        if len(label_train) > 0:
            # balanced datasets
            # g_train = label_train.groupby('label')
            # label_train = pandas.DataFrame(g_train.apply(lambda x: x.sample(g_train.size().min()).reset_index(drop=True)))
            processor = emt.data_representation.DeepMatcherProcessor()
            # trainF, validF = dm_train.tofiles(label_train, label_valid, dataset_name)
            trainF = dataset_name + '_train.csv'
            validF = dataset_name + '_valid.csv'
            label_train.to_csv(trainF)
            label_valid.to_csv(validF)
            train_examples = processor.get_train_examples_file(trainF)
            label_list = processor.get_labels()
            training_data_loader = emt.data_loader.load_data(train_examples,
                                                             label_list,
                                                             self.tokenizer,
                                                             MAX_SEQ_LENGTH,
                                                             BATCH_SIZE,
                                                             emt.data_loader.DataType.TRAINING, self.model_type)

            num_epochs = 7
            num_train_steps = len(training_data_loader) * num_epochs

            learning_rate = 2e-5
            adam_eps = 1e-8
            warmup_steps = 1
            weight_decay = 0
            optimizer, scheduler = emt.optimizer.build_optimizer(self.model,
                                                                 num_train_steps,
                                                                 learning_rate,
                                                                 adam_eps,
                                                                 warmup_steps,
                                                                 weight_decay)

            eval_examples = processor.get_test_examples_file(validF)
            evaluation_data_loader = emt.data_loader.load_data(eval_examples,
                                                               label_list,
                                                               self.tokenizer,
                                                               MAX_SEQ_LENGTH,
                                                               BATCH_SIZE,
                                                               emt.data_loader.DataType.EVALUATION, self.model_type)


            evaluation = emt.evaluation.Evaluation(evaluation_data_loader, '', exp_dir, len(label_list), self.model_type)

            result = emt.training.train(device,
                               training_data_loader,
                               self.model,
                               optimizer,
                               scheduler,
                               evaluation,
                               num_epochs,
                               1.0,
                               True,
                               experiment_name=exp_dir,
                               output_dir=exp_dir,
                               model_type=self.model_type)

        emt.model.save_model(self.model, '', exp_dir, tokenizer=self.tokenizer)
        logging.info('MODEL SAVED {}', exp_dir)
        return result

    def evaluation(self, test_set):
        device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)
        processor = emt.data_representation.DeepMatcherProcessor()
        tmpf = 'tmp.csv'
        test_set.to_csv(tmpf)
        examples = processor.get_test_examples_file(tmpf)
        test_data_loader = emt.data_loader.load_data(examples,
                                                     processor.get_labels(),
                                                     self.tokenizer,
                                                     MAX_SEQ_LENGTH,
                                                     BATCH_SIZE,
                                                     emt.data_loader.DataType.EVALUATION, self.model_type)


        evaluation = emt.evaluation.Evaluation(test_data_loader, '', '', len(test_set),
                                               self.model_type)

        result = evaluation.evaluate(self.model, device, -1)
        try:
            l0 = result.split('\n')[2].split('       ')[2].split('      ')
            l1 = result.split('\n')[3].split('       ')[2].split('      ')
        except:
            l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
            l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')

        p = l1[0]
        r = l1[1]
        f1 = l1[2]
        pnm = l0[0]
        rnm = l0[1]
        f1nm = l0[2]

        os.remove(tmpf)

        return float(p), float(r), float(f1)

    def predict(self, x, given_columns=None, mojito=False, expand_dim=False, **kwargs):
        if isinstance(x, csr_matrix):
            x = pd.DataFrame(data=np.zeros(x.shape))
            if given_columns is not None:
                x.columns = given_columns
        original = x.copy()
        if isinstance(x, np.ndarray):
            x_index = np.arange(len(x))
            xc = pd.DataFrame(x.copy(), index=x_index)
        else:
            xc = x.copy()
        if 'id' in xc.columns:
            xc = xc.drop(['id'], axis=1)
        if 'ltable_id' in xc.columns:
            xc = xc.drop(['ltable_id'], axis=1)
        if 'rtable_id' in xc.columns:
            xc = xc.drop(['rtable_id'], axis=1)
        if 'label' not in xc.columns:
            xc.insert(0, 'label', '')
        device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)
        processor = emt.data_representation.DeepMatcherProcessor()
        tmpf = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
        xc.to_csv(tmpf)
        examples = processor.get_test_examples_file(tmpf)
        test_data_loader = emt.data_loader.load_data(examples,
                                                     processor.get_labels(),
                                                     self.tokenizer,
                                                     MAX_SEQ_LENGTH,
                                                     BATCH_SIZE,
                                                     emt.data_loader.DataType.TEST, self.model_type)

        simple_accuracy, f1, classification_report, predictions = emt.prediction.predict(self.model, device,
                                                                                         test_data_loader)
        os.remove(tmpf)

        predictions.index = np.arange(len(predictions))
        if mojito:
            full_df = np.dstack((predictions['nomatch_score'], predictions['match_score'])).squeeze()
            res_shape = full_df.shape
            if len(res_shape) == 1 and expand_dim:
                res = np.expand_dims(res, axis=1).T
        else:
            names = list(xc.columns)
            names.extend(['classes', 'labels', 'nomatch_score', 'match_score'])
            xc.index = np.arange(len(xc))
            full_df = pd.concat([xc, predictions], axis=1, names=names)
            full_df.columns = names
            try:
                original.reset_index(inplace=True)
                full_df['ltable_id'] = original['ltable_id']
                full_df['rtable_id'] = original['rtable_id']
                full_df['id'] = original['id']
            except:
                pass
        return full_df

    def load(self, path):
        self.model, self.tokenizer = emt.model.load_model(path, True)
        return self

    def save(self, path):
        emt.model.save_model(self.model, path, path, tokenizer=self.tokenizer)

    def predict_proba(self, x, **kwargs):
        return self.predict(x, mojito=True, expand_dim=True)
