import os
import string
import random
import pandas as pd
import numpy as np

import emt.model
import emt.config
import emt.data_representation
import emt.data_loader
import emt.optimizer
import emt.evaluation
import emt.torch_initializer
import emt.training
import emt.prediction

BATCH_SIZE = 8

MAX_SEQ_LENGTH = 250

class EMTERModel():

    def __init__(self):
        self.model_type = 'distilbert'
        config_class, model_class, tokenizer_class = emt.config.Config().MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained('distilbert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.model = model_class.from_pretrained('distilbert-base-uncased', config=config)

    def classic_training(self, label_train, label_valid, dataset_name):
        device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)

        exp_dir = 'models/bert/' + dataset_name
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
        print(f'MODEL SAVED {exp_dir}')
        l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
        l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')
        p = l1[0]
        r = l1[1]
        f1 = l1[2]
        pnm = l0[0]
        rnm = l0[1]
        f1nm = l0[2]
        return p, r, f1, pnm, rnm, f1nm

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
        l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
        l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')
        p = l1[0]
        r = l1[1]
        f1 = l1[2]
        pnm = l0[0]
        rnm = l0[1]
        f1nm = l0[2]
        os.remove(tmpf)

        return p, r, f1, pnm, rnm, f1nm

    def predict(self, x, **kwargs):
        #print(x.head())
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

        names = list(x.columns)
        names.extend(['classes', 'labels', 'nomatch_score', 'match_score'])
        x.index = np.arange(len(x))
        predictions.index = np.arange(len(predictions))
        full_df = pd.concat([x, predictions], axis=1, names=names)
        full_df.columns = names
        print(full_df.head())
        return full_df

    def load(self, path):
        self.model, self.tokenizer = emt.model.load_model(path, True)
        return self

    def save(self, path):
        emt.model.save_model(self.model, path, path, tokenizer=self.tokenizer)
