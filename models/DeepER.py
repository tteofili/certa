import os

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import gensim.downloader as api


from models.ermodel import ERModel


def init_embeddings_index(embeddings_file):
    print('* Costruzione indice degli embeddings.....', end='', flush=True)
    embeddings_index = {}
    with open(embeddings_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f'Fatto. {len(embeddings_index)} embeddings totali.')

    return embeddings_index


# InPut: Nome del file contenente gli embeddings
# OutPut: Un modello che converte vettori di token in vettori di embeddings ed un tokenizzatore
def init_embeddings_model(embeddings_index):
    print('* Creazione del modello per il calcolo degli embeddings....', flush=True)

    print('* Inizializzo il tokenizzatore.....', end='', flush=True)
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(embeddings_index.keys())
    words_index = tokenizer.word_index
    print(f'Fatto: {len(words_index)} parole totali.')

    print('* Preparazione della matrice di embedding.....', end='', flush=True)
    embedding_dim = len(embeddings_index['cat'])  # :3
    num_words = len(words_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in words_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Le parole di cui non viene trovato l'embedding avranno vettore nullo
            embedding_matrix[i] = embedding_vector
    print(f'Fatto. Dimensioni matrice embeddings: {embedding_matrix.shape}')

    print('\n°°° EMBEDDING MODEL °°°')
    # Input layer: due tuple, ciascuna tupla è 
    # una sequenza di token (numeri)
    input_a = Input(shape=(None,), name='Tupla_A')
    input_b = Input(shape=(None,), name='Tupla_B')

    # Embedding lookup layer (shared)
    # num_words = numero di parole uniche usate in totale, 
    # embedding_dim = dimensione vettore embedding
    embedding = Embedding(num_words,
                          embedding_dim,
                          embeddings_initializer=Constant(embedding_matrix),
                          trainable=False,
                          name='Embedding_lookup')
    embedding_a = embedding(input_a)
    embedding_b = embedding(input_b)

    # Creazione modello per gli embeddings
    embeddings_model = Model(inputs=[input_a, input_b], outputs=[embedding_a, embedding_b])
    embeddings_model.summary()

    return embeddings_model, tokenizer


# OutPut: Il modello DeepER compilato pronto per l'addestramento
def init_DeepER_model(embedding_dim):
    print('\n°°° DeepER Model °°°')
    # Input layer: due sequenze di embeddings
    emb_a = Input(shape=(None, embedding_dim), name='Embeddings_seq_a')
    emb_b = Input(shape=(None, embedding_dim), name='Embeddings_seq_b')

    # Composition layer
    # Bidirectional
    lstm = Bidirectional(LSTM(150), name='Composition')
    # Unidirectional
    # lstm = LSTM(100, name='Composition')
    lstm_a = lstm(emb_a)
    lstm_b = lstm(emb_b)

    # Similarity layer (subtract or hadamart prod),
    # vedi Keras core layers e Keras merge layers per altre tipologie
    # Subtract
    # Hadamard
    similarity = Lambda(lambda ts: ts[0] * ts[1], name='Similarity')([lstm_a, lstm_b])

    # Dense layer
    dense = Dense(300, activation='relu', name='Dense1')(similarity)
    dense = Dense(300, activation='relu', name='Dense2')(dense)

    # Classification layer
    output = Dense(2, activation='softmax', name='Classification')(dense)

    # Creazione modello
    deeper_model = Model(inputs=[emb_a, emb_b], outputs=[output])

    optimizer = Adam(learning_rate=0.0001, amsgrad=True)

    # Compilazione per addestramento
    deeper_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
    deeper_model.summary()

    return deeper_model


# InPut: data = [(t1, t2, label),...], tokenizer = tokenizzatore per le tuple
# OutPut: table1, table2 = matrici di tokens, labels matrice con etichette
def data2Inputs(data, tokenizer, categorical=True):
    # Limita la sequenza massima di tokens
    # SEQUENCES_MAXLEN = 500

    # Tokenizza le tuple e prepara l'input per il modello
    print('* Preparazione input......', end='', flush=True)
    table1, table2, labels = [], [], []
    for t1, t2, label in data:
        # Sperimentale: ordino gli attributi per lunghezza decrescente
        # Attributi con molti tokens conengono più informazioni utili 
        table1.append(' '.join(t1).replace(', ', ' '))
        table2.append(' '.join(t2).replace(', ', ' '))
        labels.append(label)
    table1 = tokenizer.texts_to_sequences(table1)
    table1 = pad_sequences(table1, padding='post')
    table2 = tokenizer.texts_to_sequences(table2)
    table2 = pad_sequences(table2, padding='post')
    if categorical:
        labels = to_categorical(labels)
    else:
        labels = np.array(labels)
    print(
        f'Fatto. {len(labels)} tuple totali, esempio label: {data[0][2]} -> {labels[0]}, Table1 shape: {table1.shape}, Table2 shape: {table2.shape}')

    return table1, table2, labels


# InPut: data = [(t1, t2),...], tokenizer = tokenizzatore per le tuple
# OutPut: table1, table2 = matrici di tokens
def data2InputsUnlabel(data, tokenizer):
    # Tokenizza le tuple e prepara l'input per il modello
    table1, table2 = [], []
    for t1, t2 in data:
        # Sperimentale: ordino gli attributi per lunghezza decrescente
        # Attributi con molti tokens conengono più informazioni utili
        table1.append(' '.join(str(t1)).replace(', ', ' '))
        table2.append(' '.join(str(t2)).replace(', ', ' '))
    table1 = tokenizer.texts_to_sequences(table1)
    table1 = pad_sequences(table1, padding='post')
    table2 = tokenizer.texts_to_sequences(table2)
    table2 = pad_sequences(table2, padding='post')

    return table1, table2


# InPut: modello, nuovo output layer
# OutPut: nuovo modello con l'output layer specificato al posto di quello precedente
# I layers dovrebbero essere condivisi con il modello passato per parametro
def replace_last_layer(model, new_layer):
    x = model.layers[-2].output
    x = new_layer(x)

    return Model(inputs=model.input, outputs=x)


# InPut: Una lista di triple [(tup1, tup2, label), ...], il modello da addestrare...
# Output: Il modello addestrato
def train_model_ER(data, valid, model, embeddings_model, tokenizer, pretraining=False, metric='val_mse', end='',
                   save_path=None):
    if pretraining:
        model_name = 'VinSim'
        table1, table2, labels = data2Inputs(data, tokenizer, categorical=False)
        vtable1, vtable2, vlabels = data2Inputs(valid, tokenizer, categorical=False)
    else:
        model_name = 'DeepER'
        table1, table2, labels = data2Inputs(data, tokenizer)
        vtable1, vtable2, vlabels = data2Inputs(valid, tokenizer)

    # Preparazione embeddings (tokens -> embeddings)
    x1, x2 = embeddings_model.predict([table1, table2])
    v1, v2 = embeddings_model.predict([vtable1, vtable2])

    # Early stopping (arresta l'apprendimento se non ci sono miglioramenti)
    es = EarlyStopping(monitor=metric, min_delta=0, verbose=1, patience=7)
    # Model checkpointing (salva il miglior modello fin'ora)
    mc = ModelCheckpoint(f'models/saved/deeper/{model_name}_best_model_{end}.h5', monitor=metric, verbose=1,
                         save_best_only=True)

    # Addestramento modello
    param_batch_size = 16
    print('Batch size:', param_batch_size)
    model.fit([x1, x2], labels, batch_size=param_batch_size, epochs=64, validation_data=([v1, v2], vlabels),
              callbacks=[es, mc])

    # Carica il miglior modello checkpointed
    # model = load_model(f'{model_name}_best_model{end}.h5')
    if save_path is not None:
        save(model, save_path)
    return model


def save(model, path):
    model.save(path)


def train_model_ROUND_ER(data, model, embeddings_model, tokenizer, metric='val_accuracy', end=''):
    model_name = 'VinSim'

    table1, table2, labels = data2Inputs(data, tokenizer)

    # Preparazione embeddings (tokens -> embeddings)
    x1, x2 = embeddings_model.predict([table1, table2])

    # Early stopping (arresta l'apprendimento se non ci sono miglioramenti)
    es = EarlyStopping(monitor=metric, min_delta=0, verbose=1, patience=7)
    # Model checkpointing (salva il miglior modello fin'ora)
    mc = ModelCheckpoint(f'{model_name}_best_model{end}.h5', monitor=metric, verbose=1, save_best_only=True)
    # Addestramento modello
    param_batch_size = round(len(data) * 0.015) + 1
    print('Batch size:', param_batch_size)
    model.fit([x1, x2], labels, batch_size=param_batch_size, epochs=64, validation_split=0.2, callbacks=[es, mc])

    # Carica il miglior modello checkpointed
    model = load_model(f'{model_name}_best_model{end}.h5')

    return model


# F-Measure
# InPut: Dati nel formato [(tupla1, tupla2, label),...], un modello da testare
# OutPut: Statistiche sul modello 
def model_statistics(data, model, embeddings_model, tokenizer):
    print('* Avvio test metriche....', flush=True)

    corpus_size = len(data)

    no_match_count = 0
    match_count = 0
    for tri in data:
        if tri[2] == 1:
            match_count += 1
        else:
            no_match_count += 1

    print(f'-- Corpus size: {corpus_size}')
    print(f'-- Non Match: {no_match_count}')
    print(f'-- Match: {match_count}')

    # Crea matrici di tokens e labels
    table1, table2, _ = data2Inputs(data, tokenizer)

    # Calcola inputs di embeddings
    emb1, emb2 = embeddings_model.predict([table1, table2])

    match_retrieved = []

    true_match = 0

    print('* Evaluating: ', end='', flush=True)

    pred_matrix = model.predict([emb1, emb2])

    for i in range(corpus_size):
        prediction = pred_matrix[i]
        if np.argmax(prediction) == 1:
            match_retrieved.append(data[i])
            # Conta predizioni di match corrette
            if data[i][2] == 1:
                true_match += 1
        if i % (corpus_size // 2) == 0:
            print(f'=', end='', flush=True)

    print('|')

    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(match_retrieved)
        precision = true_match / retrieved_size
        recall = true_match / match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print(f'Precision: {precision}, Recall: {recall}, f1-score: {fmeasure}')
        print(f'Total retrieved: {retrieved_size}, retrieved/total matches: {true_match}/{match_count}')

    except:
        print(f'Error. Retrieved = {retrieved_size}, Matches = {match_count} ')
        precision = -1
        recall = -1

    return (precision, recall, fmeasure)


def predict(data, model, embeddings_model, tokenizer):
    # Crea matrici di tokens e labels
    table1, table2 = data2InputsUnlabel(data, tokenizer)

    # Calcola inputs di embeddings
    emb1, emb2 = embeddings_model.predict([table1, table2])

    pred_matrix = model.predict([emb1, emb2])

    return pred_matrix


# F-Measure
# InPut: Dati nel formato [(tupla1, tupla2, label),...], un modello da testare
# OutPut: Statistiche sul modello 
def model_statistics_prf(data, model, embeddings_model, tokenizer):
    print('* Avvio test metriche....', flush=True)

    corpus_size = len(data)

    no_match_count = 0
    match_count = 0
    for tri in data:
        if int(tri[2]) == 1:
            match_count += 1
        else:
            no_match_count += 1

    print('-- Corpus size: {a}'.format(a=corpus_size))
    print('-- Non Match: {a}'.format(a=no_match_count))
    print('-- Match: {a}'.format(a=match_count))

    # Crea matrici di tokens e labels
    table1, table2, _ = data2Inputs(data, tokenizer)

    # Calcola inputs di embeddings
    emb1, emb2 = embeddings_model.predict([table1, table2])

    match_retrieved = []

    true_match = 0

    print('* Evaluating: ', end='', flush=True)

    pred_matrix = model.predict([emb1, emb2])

    for i in range(corpus_size):
        prediction = pred_matrix[i]
        if np.argmax(prediction) == 1:
            match_retrieved.append(data[i])
            # Conta predizioni di match corrette
            if int(data[i][2]) == 1:
                true_match += 1
        # if i % (corpus_size // 10) == 0:
        #     print('=', end='', flush=True)

    print('|')

    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(match_retrieved)
        precision = true_match / retrieved_size
        recall = true_match / match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print('Precision: {a}, Recall: {b}, f1-score: {c}'.format(a=precision, b=recall, c=fmeasure))
        print('Total retrieved: {a}, retrieved/total matches: {b}/{c}'.format(a=retrieved_size, b=true_match,
                                                                              c=match_count))

    except:

        print('Error. Retrieved = {a}, Matches = {b} '.format(a=retrieved_size, b=match_count))
        precision = -1
        recall = -1

    return precision, recall, fmeasure


# F-MeasureNOmatch
# InPut: Dati nel formato [(tupla1, tupla2, label),...], un modello da testare
# OutPut: Statistiche sul modello 
def model_statisticsNOMatch_prf(data, model, embeddings_model, tokenizer):
    print('* Avvio test metriche....', flush=True)

    corpus_size = len(data)

    no_match_count = 0
    match_count = 0
    for tri in data:
        if tri[2] == 1:
            match_count += 1
        else:
            no_match_count += 1

    print('-- Corpus size: {a}'.format(a=corpus_size))
    print('-- Non Match: {a}'.format(a=no_match_count))
    print('-- Match: {a}'.format(a=match_count))

    # Crea matrici di tokens e labels
    table1, table2, _ = data2Inputs(data, tokenizer)

    # Calcola inputs di embeddings
    emb1, emb2 = embeddings_model.predict([table1, table2])

    no_match_retrieved = []

    true_NOmatch = 0

    print('* Evaluating: ', end='', flush=True)

    pred_matrix = model.predict([emb1, emb2])

    for i in range(corpus_size):
        prediction = pred_matrix[i]
        if np.argmax(prediction) == 1:
            no_match_retrieved.append(data[i])
            # Conta predizioni di no_match corrette
            if data[i][2] == 0:
                true_NOmatch += 1
        # if i % (corpus_size // 10) == 0:
        #     print('=', end='', flush=True)

    print('|')

    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(no_match_retrieved)
        precision = true_NOmatch / retrieved_size
        recall = true_NOmatch / no_match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print('Precision: {a}, Recall: {b}, f1-score: {c}'.format(a=precision, b=recall, c=fmeasure))
        print('Total retrieved: {a}, retrieved/total no_matches: {b}/{c}'.format(a=retrieved_size, b=true_NOmatch,
                                                                                 c=no_match_count))

    except:

        print('Error. Retrieved = {a}, Matches = {b} '.format(a=retrieved_size, b=match_count))
        precision = -1
        recall = -1

    return precision, recall, fmeasure


def deeper_mojito_predict(predict_fn):
    def wrapper(dataframe):
        dataframe['id'] = np.arange(len(dataframe))
        output = predict_fn(dataframe)
        return np.dstack((output['nomatch_score'], output['match_score'])).squeeze()

    return wrapper


class DeepERModel(ERModel):

    def __init__(self):

        super(DeepERModel, self).__init__()
        self.name = 'deeper'
        if not os.path.exists('models/glove.6B.300d.txt'):
            word_vectors = api.load("glove-wiki-gigaword-300")
            word_vectors.save_word2vec_format('models/glove.6B.300d.txt', binary=False)

        self.embeddings_index = init_embeddings_index('models/glove.6B.300d.txt')
        emb_dim = len(self.embeddings_index['cat'])

        self.embeddings_model, self.tokenizer = init_embeddings_model(self.embeddings_index)

        self.model = init_DeepER_model(emb_dim)

    def train(self, label_train_df, label_valid_df, DATASET_NAME):

        label_train = to_deeper_data(label_train_df)
        label_valid = to_deeper_data(label_valid_df)
        # sub_data = label_train
        perc = len(label_train)

        average_arch = False

        if perc != 0:
            self.model = train_model_ER(label_train,
                                        label_valid,
                                        self.model,
                                        self.embeddings_model,
                                        self.tokenizer,
                                        pretraining=False,
                                        end='_{}_{}'.format(int(perc), DATASET_NAME))

        return self.evaluation(label_valid_df)  # self.model #.run_eval(label_valid)

    def evaluation(self, test_set_df):

        try:
            test_set_df = test_set_df.drop(['ltable_id', 'rtable_id'])
        except:
            pass

        test_set = to_deeper_data(test_set_df)

        precision, recall, fmeasure = model_statistics_prf(test_set, self.model, self.embeddings_model,
                                                           self.tokenizer)

        precisionNOMATCH, recallNOMATCH, fmeasureNOMATCH = model_statisticsNOMatch_prf(test_set, self.model,
                                                                                       self.embeddings_model,
                                                                                       self.tokenizer)

        return precision, recall, fmeasure

    def predict(self, x, given_columns=None, mojito=False, expand_dim=False, ignore_columns=[], **kwargs):
        if isinstance(x, csr_matrix):
            x = pd.DataFrame(data=np.zeros(x.shape))
            if given_columns is not None:
                x.columns = given_columns
        if isinstance(x, np.ndarray):
            data = to_deeper_data_np(x)
            x_index = np.arange(len(x))
            x_copy = pd.DataFrame(index=x_index)
        else:
            data = to_deeper_data(x.drop([c for c in ignore_columns if c in x.columns], axis=1))
            x_index = x.index
            x_copy = x.copy()
        out = predict(data, self.model, self.embeddings_model, self.tokenizer)
        out_df = pd.DataFrame(out, columns=['nomatch_score', 'match_score'])
        out_df.index = x_index
        res = pd.concat([x_copy, out_df], axis=1)
        if mojito:
            res = np.dstack((res['nomatch_score'], res['match_score'])).squeeze()
            res_shape = res.shape
            if len(res_shape) == 1 and expand_dim:
                res = np.expand_dims(res, axis=1).T
        return res

    def save(self, path):
        save(self.model, path)

    def load(self, path):
        self.model = load_model(path)
        return self

    def predict_proba(self, x, **kwargs):
        return self.predict(x, mojito=True, expand_dim=True)


def to_deeper_data(df: pd.DataFrame):
    res = []
    for r in range(len(df)):
        row = df.iloc[r]
        lpd = row.filter(regex='^ltable_')
        rpd = row.filter(regex='^rtable_')
        if 'label' in row:
            label = row['label']
            res.append((lpd.values.astype('str'), rpd.values.astype('str'), label))
        else:
            res.append((lpd.values.astype('str'), rpd.values.astype('str')))
    return res


def to_deeper_data_np(array: np.array):
    res = []
    if array.shape[1] % 2 != 0:
        columns = (array.shape[1] - 1) // 2 + 1
        start = 1
    else:
        columns = (array.shape[1]) // 2 + 1
        start = 0
    for r in range(len(array)):
        row = array[r]
        lpd = row[start:columns]
        rpd = row[columns:]
        res.append((lpd, rpd))
    return res
