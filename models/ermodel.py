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

