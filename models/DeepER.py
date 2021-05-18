import numpy as np
from keras.layers import Input, Embedding, LSTM, concatenate, subtract, Dense, Bidirectional, Lambda
from keras.models import Model, load_model
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

# InPut: Nome del file con gli embeddings
# Output: Un dizionario con tutti gli embeddings: {parola: embedding}
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
    embedding_dim = len(embeddings_index['cat']) # :3  
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
    #lstm = LSTM(100, name='Composition')
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

    # Compilazione per addestramento
    deeper_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #deeper_model.summary()

    return deeper_model


# InPut: data = [(t1, t2, label),...], tokenizer = tokenizzatore per le tuple
# OutPut: table1, table2 = matrici di tokens, labels matrice con etichette
def data2Inputs(data, tokenizer, categorical=True):
    
    # Limita la sequenza massima di tokens 
    #SEQUENCES_MAXLEN = 500

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
    print(f'Fatto. {len(labels)} tuple totali, esempio label: {data[0][2]} -> {labels[0]}, Table1 shape: {table1.shape}, Table2 shape: {table2.shape}')

    return table1, table2, labels


# InPut: data = [(t1, t2),...], tokenizer = tokenizzatore per le tuple
# OutPut: table1, table2 = matrici di tokens
def data2InputsUnlabel(data, tokenizer):
    # Tokenizza le tuple e prepara l'input per il modello
    table1, table2 = [], []
    for t1, t2 in data:
        # Sperimentale: ordino gli attributi per lunghezza decrescente
        # Attributi con molti tokens conengono più informazioni utili
        table1.append(' '.join(t1).replace(', ', ' '))
        table2.append(' '.join(t2).replace(', ', ' '))
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
def train_model_ER(data, model, embeddings_model, tokenizer, pretraining=False, metric='val_accuracy', end='',
                   save_path=None):

    if pretraining:
        model_name = 'VinSim'
        table1, table2, labels = data2Inputs(data, tokenizer, categorical=False)
    else:
        model_name = 'DeepER'
        table1, table2, labels = data2Inputs(data, tokenizer)         
    
    # Preparazione embeddings (tokens -> embeddings)
    x1, x2 = embeddings_model.predict([table1, table2])        
    
    # Early stopping (arresta l'apprendimento se non ci sono miglioramenti)
    es = EarlyStopping(monitor=metric, min_delta=0, verbose=1, patience=7)
    # Model checkpointing (salva il miglior modello fin'ora)
    mc = ModelCheckpoint(f'models/deeper/{model_name}_best_model_{end}.h5', monitor=metric, verbose=1, save_best_only=True)
    # Addestramento modello
    param_batch_size = round(len(data) * 0.015) + 1
    print('Batch size:', param_batch_size)
    model.fit([x1, x2], labels, batch_size=param_batch_size, epochs=64, validation_split=0.2, callbacks=[es, mc])

    # Carica il miglior modello checkpointed
    #model = load_model(f'{model_name}_best_model{end}.h5')
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
        if i % (corpus_size // 10) == 0:
            print(f'=', end='', flush=True)

    print('|')
    
    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(match_retrieved)
        precision = true_match / retrieved_size
        recall =  true_match /  match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print(f'Precision: {precision}, Recall: {recall}, f1-score: {fmeasure}')
        print(f'Total retrieved: {retrieved_size}, retrieved/total matches: {true_match}/{match_count}')

    except:
        print(f'Error. Retrieved = {retrieved_size}, Matches = {match_count} ')
        precision=-1
        recall=-1

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
        if i % (corpus_size // 10) == 0:
            print('=', end='', flush=True)

    print('|')
    
    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(match_retrieved)
        precision = true_match / retrieved_size
        recall =  true_match /  match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print('Precision: {a}, Recall: {b}, f1-score: {c}'.format(a=precision,b=recall,c=fmeasure))
        print('Total retrieved: {a}, retrieved/total matches: {b}/{c}'.format(a=retrieved_size,b=true_match,c=match_count))

    except:

        print('Error. Retrieved = {a}, Matches = {b} '.format(a=retrieved_size,b=match_count))
        precision=-1
        recall=-1

    return precision,recall,fmeasure


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
        if i % (corpus_size // 10) == 0:
            print('=', end='', flush=True)

    print('|')
    
    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(no_match_retrieved)
        precision = true_NOmatch / retrieved_size
        recall =  true_NOmatch /  no_match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print('Precision: {a}, Recall: {b}, f1-score: {c}'.format(a=precision,b=recall,c=fmeasure))
        print('Total retrieved: {a}, retrieved/total no_matches: {b}/{c}'.format(a=retrieved_size,b=true_NOmatch,c=no_match_count))

    except:

        print('Error. Retrieved = {a}, Matches = {b} '.format(a=retrieved_size,b=match_count))
        precision=-1
        recall=-1

    return precision,recall,fmeasure
