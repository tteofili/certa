import os
import string
import random

import pandas

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

MAX_SEQ_LENGTH = 128

def split_training_valid(pt_train, SPLIT_FACTOR=0.8):
    bound = int(len(pt_train) * SPLIT_FACTOR)
    train = pt_train[:bound]
    valid = pt_train[bound:]

    return train, valid


class ERModel:

    def init(self):
        self.name = ''
        pass

    def predict(self, x, **kwargs):
        pass

    def train(self, label_train, label_valid, dataset_name):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def evaluation(self, test_set):
        pass

#
# class DMERModel(ERModel):
#
#     def __init__(self):
#         super(DMERModel, self).__init__()
#         self.model = dm.MatchingModel(attr_summarizer='hybrid')
#
#     def initialize_models(self, data):
#         self.model.initialize(data)
#
#     def train(self, label_train, label_valid, dataset_name):
#         train_file = dataset_name+'_dm_train.csv'
#         valid_file = dataset_name+'_dm_valid.csv'
#         label_train.to_csv(train_file, index=False)
#         label_valid.to_csv(valid_file, index=False)
#
#
#         # read dataset
#         trainLab, validationLab = dm.data.process(cache=dataset_name+'.pth', path='', train=train_file, validation=valid_file, left_prefix='ltable_',
#             right_prefix='rtable_',)
#         self.initialize_models(trainLab)
#
#         print("CLASSIC TRAINING with " + str(len(trainLab)) + " samples")
#         # train default model with standard dataset
#         self.model.run_train(trainLab, validationLab, best_save_path=dataset_name + '_best_default_model.pth')
#
#         stats = self.model.run_eval(validationLab)
#
#         pm = stats.precision()
#         rm = stats.recall()
#         pnm = stats.precisionNM()
#         rnm = stats.recallNM()
#         f1nm = stats.f1NM()
#         f1m = stats.f1()
#         return pm, rm, f1m, pnm, rnm, f1nm
#
#     def predict(self, x, **kwargs):
#         xc = x.copy()
#         # if 'id' in xc.columns:
#         #     xc = xc.drop(['id'], axis=1)
#         return wrapDm(xc, self.model, **kwargs)
#
#     def load(self, path):
#         self.model.load_state(path)
#
#     def save(self, path):
#         self.model.save_state(path, True)
#
#
# class DeeperERModel(ERModel):
#
#     def __init__(self):
#
#         super(DeeperERModel, self).__init__()
#         self.embeddings_index = dp.init_embeddings_index('glove.6B.50d.txt')
#         emb_dim = len(self.embeddings_index['cat'])  # :3
#
#         average_arch = False
#         if average_arch == False:
#             self.embeddings_model, self.tokenizer = dp.init_embeddings_model(self.embeddings_index)
#         else:
#             self.embeddings_model, self.tokenizer = dp.init_embeddings_modelAverage(self.embeddings_index)
#
#         self.model = dp.init_DeepER_model(emb_dim)
#
#     def train(self, label_train, label_valid, DATASET_NAME):
#         # sub_data = label_train
#         perc = len(label_train)
#
#         average_arch = False
#
#         if perc != 0:
#             if average_arch == False:
#                 self.model = dp.train_model_ER(label_train,
#                                                self.model,
#                                                self.embeddings_model,
#                                                self.tokenizer,
#                                                pretraining=False,
#                                                end='_{}_{}'.format(int(perc), DATASET_NAME))
#             else:
#                 self.model = dp.train_model_ER_PREAverage(label_train,
#                                                           self.model,
#                                                           self.embeddings_model,
#                                                           self.tokenizer,
#                                                           pretraining=False,
#                                                           end='_{}_{}'.format(int(perc), DATASET_NAME))
#
#         return self.evaluation(label_valid)  # self.model #.run_eval(label_valid)
#
#     def evaluation(self, test_set):
#
#         precision, recall, fmeasure = dp.model_statistics_prf(test_set, self.model, self.embeddings_model,
#                                                               self.tokenizer)
#
#         precisionNOMATCH, recallNOMATCH, fmeasureNOMATCH = dp.model_statisticsNOMatch_prf(test_set, self.model,
#                                                                                           self.embeddings_model,
#                                                                                           self.tokenizer)
#
#         return precision, recall, fmeasure, precisionNOMATCH, recallNOMATCH, fmeasureNOMATCH
#
#     def predict(self, x, **kwargs):
#         return dp.predict(x, self.model, self.embeddings_model, self.tokenizer)
#
#     def save(self, path):
#         dp.save(self.model, path)
#
#     def load(self, path):
#         return dp.load_model(path)
#
#
# class EMTERModel(ERModel):
#
#     def __init__(self):
#         self.model_type = 'distilbert'
#         super(EMTERModel, self).__init__()
#         config_class, model_class, tokenizer_class = emt.config.Config().MODEL_CLASSES[self.model_type]
#         config = config_class.from_pretrained('distilbert-base-uncased')
#         self.tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
#         self.model = model_class.from_pretrained('distilbert-base-uncased', config=config)
#
#     def train(self, label_train, label_valid, dataset_name):
#         device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)
#
#         exp_dir = 'models/emt/' + dataset_name
#         if len(label_train) > 0:
#             # balanced datasets
#             # g_train = label_train.groupby('label')
#             # label_train = pandas.DataFrame(g_train.apply(lambda x: x.sample(g_train.size().min()).reset_index(drop=True)))
#             processor = emt.data_representation.DeepMatcherProcessor()
#             # trainF, validF = dm_train.tofiles(label_train, label_valid, dataset_name)
#             trainF = dataset_name + '_train.csv'
#             validF = dataset_name + '_valid.csv'
#             label_train.to_csv(trainF)
#             label_valid.to_csv(validF)
#             train_examples = processor.get_train_examples_file(trainF)
#             label_list = processor.get_labels()
#             training_data_loader = emt.data_loader.load_data(train_examples,
#                                                              label_list,
#                                                              self.tokenizer,
#                                                              MAX_SEQ_LENGTH,
#                                                              BATCH_SIZE,
#                                                              emt.data_loader.DataType.TRAINING, self.model_type)
#
#             num_epochs = 5
#             num_train_steps = len(training_data_loader) * num_epochs
#
#             learning_rate = 2e-5
#             adam_eps = 1e-8
#             warmup_steps = 1
#             weight_decay = 0
#             optimizer, scheduler = emt.optimizer.build_optimizer(self.model,
#                                                                  num_train_steps,
#                                                                  learning_rate,
#                                                                  adam_eps,
#                                                                  warmup_steps,
#                                                                  weight_decay)
#
#             eval_examples = processor.get_test_examples_file(validF)
#             evaluation_data_loader = emt.data_loader.load_data(eval_examples,
#                                                                label_list,
#                                                                self.tokenizer,
#                                                                MAX_SEQ_LENGTH,
#                                                                BATCH_SIZE,
#                                                                emt.data_loader.DataType.EVALUATION, self.model_type)
#
#
#             evaluation = emt.evaluation.Evaluation(evaluation_data_loader, '', exp_dir, len(label_list), self.model_type)
#
#             result = emt.training.train(device,
#                                training_data_loader,
#                                self.model,
#                                optimizer,
#                                scheduler,
#                                evaluation,
#                                num_epochs,
#                                1.0,
#                                True,
#                                experiment_name=exp_dir,
#                                output_dir=exp_dir,
#                                model_type=self.model_type)
#
#         emt.model.save_model(self.model, '', exp_dir, tokenizer=self.tokenizer)
#         print(f'MODEL SAVED {exp_dir}')
#         l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
#         l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')
#         p = l1[0]
#         r = l1[1]
#         f1 = l1[2]
#         pnm = l0[0]
#         rnm = l0[1]
#         f1nm = l0[2]
#         return p, r, f1, pnm, rnm, f1nm
#
#     def evaluation(self, test_set):
#         device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)
#         processor = emt.data_representation.DeepMatcherProcessor()
#         tmpf = 'tmp.csv'
#         test_set.to_csv(tmpf)
#         examples = processor.get_test_examples_file(tmpf)
#         test_data_loader = emt.data_loader.load_data(examples,
#                                                      processor.get_labels(),
#                                                      self.tokenizer,
#                                                      MAX_SEQ_LENGTH,
#                                                      BATCH_SIZE,
#                                                      emt.data_loader.DataType.EVALUATION, self.model_type)
#
#
#         evaluation = emt.evaluation.Evaluation(test_data_loader, '', '', len(test_set),
#                                                self.model_type)
#
#         result = evaluation.evaluate(self.model, device, -1)
#         l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
#         l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')
#         p = l1[0]
#         r = l1[1]
#         f1 = l1[2]
#         pnm = l0[0]
#         rnm = l0[1]
#         f1nm = l0[2]
#         os.remove(tmpf)
#
#         return p, r, f1, pnm, rnm, f1nm
#
#     def predict(self, x, **kwargs):
#         #print(x.head())
#         xc = x.copy()
#         if 'id' in xc.columns:
#             xc = xc.drop(['id'], axis=1)
#         if 'ltable_id' in xc.columns:
#             xc = xc.drop(['ltable_id'], axis=1)
#         if 'rtable_id' in xc.columns:
#             xc = xc.drop(['rtable_id'], axis=1)
#         if 'label' not in xc.columns:
#             xc.insert(0, 'label', '')
#         device, n_gpu = emt.torch_initializer.initialize_gpu_seed(22)
#         processor = emt.data_representation.DeepMatcherProcessor()
#         tmpf = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
#         xc.to_csv(tmpf)
#         examples = processor.get_test_examples_file(tmpf)
#         test_data_loader = emt.data_loader.load_data(examples,
#                                                      processor.get_labels(),
#                                                      self.tokenizer,
#                                                      MAX_SEQ_LENGTH,
#                                                      BATCH_SIZE,
#                                                      emt.data_loader.DataType.TEST, self.model_type)
#
#         simple_accuracy, f1, classification_report, predictions = emt.prediction.predict(self.model, device,
#                                                                                          test_data_loader)
#         os.remove(tmpf)
#         return predictions
#
#     def load(self, path):
#         self.model, self.tokenizer = emt.model.load_model(path, True)
#         return self
#
#     def save(self, path):
#         emt.model.save_model(self.model, path, path, tokenizer=self.tokenizer)
