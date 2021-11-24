class ERModel:
    '''
    Interface for Entity Resolution models to be explained
    '''

    def __init__(self):
        self.name = ''

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

    def predict_proba(self, x, **kwargs):
        pass

