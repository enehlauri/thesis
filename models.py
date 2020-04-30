from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.layers import Conv2D, LSTM, Conv1D, MaxPooling2D
from tensorflow.keras.layers import Input, Activation, RepeatVector, Reshape
from tensorflow.keras.layers import multiply, Permute, GlobalAveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from keras.callbacks.callbacks import EarlyStopping
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy

def print_result(name, Y_true, Y_pred):
    p = precision_score(Y_true, Y_pred,average='macro') 
    r = recall_score(Y_true, Y_pred,average='macro')
    f1 = f1_score(Y_true, Y_pred,average='macro')
    kappa = cohen_kappa_score(Y_true, Y_pred)
    
    print('\nResults for', name)
    print('Accuracy: ', accuracy_score(Y_true, Y_pred))
    print('Precision: {:.3f} \nRecall: {:.3f} \
          \nF1 on test: {:.3f} \nCohens Kappa {:.3f}'.format(p, r, f1, kappa))
    print('Confusion Matrix: \n', confusion_matrix(Y_true, Y_pred))
    print()
    return p, r, f1, kappa

def get_f1(y_true, y_pred): 
    #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def benchmark(data):  
    # Benchmark predictions for test data
    
    Y_true = np.zeros([test_lenght*N_TOP_ACTIVE, 1])
    pos = 0
    
    # Get ground truth
    for i in range(N_TOP_ACTIVE):
        for j in range(data.shape[0]-test_lenght, data.shape[0]):
            Y_true[pos] = data[j, i]
            pos+=1
            
    # Initialize prediction as 0 and change to 1 after -1 and -1 after 1
    bench_pred = np.zeros([test_lenght*N_TOP_ACTIVE, 1])
    ind = 0
    while ind < Y_true.shape[0]-1:
        if Y_true[ind] != 0:
            bench_pred[ind+1] = -1*Y_true[ind]
        ind +=1
    
    # Metrics for benchmark prediction
    print_result('Benchmark', Y_true, bench_pred)
    
    # Metrics for just zeros benchmark
    z_pred = np.zeros([test_lenght*N_TOP_ACTIVE, 1])
    print_result('Just zeros', Y_true, z_pred)
    
    return 

def split_train_test(X, Y):
    # Split formatted data into train and test. First train_period days for
    # training and the rest for testing
    
    features = 13
    days = X.shape[1]
    
    train_period = int(X.shape[0]/N_TOP_ACTIVE - test_lenght)
    
    X_train = np.zeros([train_period*N_TOP_ACTIVE, days, features])    
    Y_train = np.zeros([train_period*N_TOP_ACTIVE, Y.shape[1]])
    
    X_test = np.zeros([test_lenght*N_TOP_ACTIVE, days, features])    
    Y_test = np.zeros([test_lenght*N_TOP_ACTIVE, Y.shape[1]])
    
    # Iterate through X and Y and place train_period samples for training and 
    # next 250 for testing
    i = 0
    to_train_set=True
    nth_time = 0
    while i < X.shape[0]:        
        if to_train_set:
            X_train[nth_time*train_period:(nth_time+1)*train_period,:] =\
                X[i:i+train_period,:]
            Y_train[nth_time*train_period:(nth_time+1)*train_period,:] =\
                Y[i:i+train_period,:]
            i += train_period
            to_train_set=False
        else:
            X_test[nth_time*test_lenght:(nth_time+1)*test_lenght,:] = \
                X[i:i+test_lenght,:]
            Y_test[nth_time*test_lenght:(nth_time+1)*test_lenght,:] = \
                Y[i:i+test_lenght,:]
            i += test_lenght
            to_train_set=True
            nth_time += 1
    
    return X_train, X_test, Y_train, Y_test

def format_data(data, days):
    # Format data to sequences of lenght days
    
    features = 13
    
    X = np.zeros([(data.shape[0] -days)*N_TOP_ACTIVE, days, features])    
    Y = np.zeros([(data.shape[0] -days)*N_TOP_ACTIVE, 1])
    
    pos = 0
    for i in range(N_TOP_ACTIVE):
        for j in range(data.shape[0] - days):
            # Investor i activity, eurovolume and time since last trade
            matrix = data[j:days+j,(i, i+N_TOP_ACTIVE, i+2*N_TOP_ACTIVE)] 
            
            # Number of active investors and market data
            matrix = np.hstack([matrix, data[j:days+j, -10:]])
            
            X[pos] = copy.deepcopy(matrix)
            Y[pos] = data[j +days, i]
            pos += 1
    print('Input and output prepared')
    return X, Y

def attention_layer(inputs, length, width):
    # From mäkinen et al. 2019
    att_weights = Dense(1, activation='tanh')(inputs)
    att_weights = Flatten()(att_weights)
    att_weights = Activation('softmax')(att_weights)
    att_weights = RepeatVector(width)(att_weights)
    att_weights = Permute([2, 1])(att_weights)
    weighted_input = multiply([inputs, att_weights])
    return weighted_input



def mlp_model(input_shape, neurons, net_width, net_depth, attention=False):
    
    input_height = input_shape[1]
    base_neurons = neurons

    inputs = Input(shape=(input_height))
    
    if attention:
        layer = attention_layer(inputs, input_height, 1)
    else:
        layer = inputs
    
    for i in range(net_depth):
        
        layer = Dense(base_neurons*net_width, activation='softmax')(layer)
        layer = Dropout(0.5)(layer)
        
    outputs = Dense(num_classes, activation='softmax')(layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[]) #'accuracy', get_f1. Per class accuracy?
    return model

def train_and_testMLP(data):
    
    neurons = 15
    
    X, Y = format_data(data, 1)
    Y = to_categorical(Y, num_classes)
    
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    
    X_train = X_train[:,-1,:]
    X_test = X_test[:,-1,:]
    
    Y_true = np.argmax(Y_test, axis=1) 
    results = list()
    
    for net_width in [1, 2, 3]:
        for net_depth in [1, 2, 3]:
            
                model_mlp = mlp_model(X_train.shape, neurons,
                                      net_width, net_depth)
    
                model_mlp.fit(X_train, Y_train, epochs=10, verbose=1,
                              batch_size=16, class_weight=class_weight)
                
                # Will work on walk forward validation next
                # Testing and reporting results
                prob = model_mlp.predict(X_test)
                Y_pred = np.argmax(prob, axis=1)
                
                name = 'MLP '+ str(net_width) + '-' + str(net_depth)
                p, r, f1, kappa = print_result(name, Y_true, Y_pred)
                results.append([name, p, r, f1, kappa])
                
    for row in results:
        print(row)
    
    return model_mlp


def lstm_model(input_shape, neurons, net_width, net_depth, attention=False):
    time_steps = input_shape[1]
    features = input_shape[2]
    
    base_neurons = neurons
    
    inputs = Input(shape=(time_steps, features))
    
    if attention:
        layer = attention_layer(inputs, time_steps, features)
    else:
        layer = inputs
        
    for i in range(net_depth):
        # return sequences True, excpet on the last layer
        return_sequences = True if not (i+1 == net_depth) else False
            
        layer = LSTM(base_neurons*net_width, activation='softmax', 
                     return_sequences=return_sequences)(layer)
        layer = Dropout(0.5)(layer)
    
    outputs = Dense(num_classes, activation='sigmoid', dynamic=True)(layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy', get_f1]) 
    return model

def train_and_testLSTM(data):   
    
    window = 10
    neurons = 15
            
    X, Y = format_data(data, window)
    Y = to_categorical(Y, num_classes)
    
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
           
    Y_true = np.argmax(Y_test, axis=1) 
    results = list()
    
    for net_width in [1, 2, 3]:
        for net_depth in [1, 2]:
                model_lstm = lstm_model(X_train.shape, neurons, 
                                        net_width, net_depth)
    
                model_lstm.fit(X_train, Y_train, epochs=6, verbose=1,
                              batch_size=16, class_weight=class_weight)
                
                
                # Will work on walk forward validation next
                # Testing and reporting results
                prob = model_lstm.predict(X_test)
                Y_pred = np.argmax(prob, axis=1)
    
                name = 'LSTM '+ str(net_width) + '-' + str(net_depth)
                p, r, f1, kappa = print_result(name, Y_true, Y_pred)
                results.append([name, p, r, f1, kappa])
                
    for row in results:
        print(row)
        
    return model_lstm

def cnn_model(input_shape, net_width, net_depth, attention=False):
    
    conv_size = 4
    filters = 32
    
    time_steps = input_shape[1]
    features = input_shape[2]
    
    # Build network
    inputs = Input(shape=(time_steps, features))
    layer = inputs

    for i in range(net_depth):
        layer = Conv1D(filters*net_width, conv_size, activation='relu')(layer)
        layer = Conv1D(filters*net_width, conv_size, activation='relu')(layer)
        
        if i+1 != net_depth:
            layer = MaxPooling1D(2)(layer)
        else: 
            layer = GlobalAveragePooling1D()(layer)
            
    dropout = Dropout(0.25)(layer)        
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[]) #'accuracy', get_f1
    
    return model

def train_and_testCNN(data):
    window = 22
    
    X, Y = format_data(data, window)
    Y = to_categorical(Y, num_classes)
    
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    
    # Weights?
    
    Y_true = np.reshape(Y_test, (Y_test.shape[0], num_classes))
    Y_true = np.argmax(Y_true, axis=1)
    results = list()

    for net_width in [3]:
        for net_depth in [2]:
            model_cnn = cnn_model(X_train.shape, net_width, net_depth)
    
            model_cnn.fit(X_train, Y_train, batch_size = 32, epochs=20)
            
            # Will work on walk forward validation next
            # Testing and reporting results
            prob = model_cnn.predict(X_test)
            prob = np.reshape(prob, (prob.shape[0], num_classes))
            Y_pred = np.argmax(prob, axis=1)
            
            name = 'CNN '+ str(net_width) + '-' + str(net_depth)
            p, r, f1, kappa = print_result(name, Y_true, Y_pred)
            results.append([name, p, r, f1, kappa])
            
    for row in results:
        print(row)
        

    return model_cnn


loc = 'C:\\Users\\enehl\\OneDrive - TUNI.fi\\Opiskelu\\Kandi\\Code'  
data = np.load(loc + '\\data.npy')

N_TOP_ACTIVE = 100
num_classes = 3
test_lenght = 250


weight = 5.

class_weight = {0: 1.,
                1: weight,
                2: weight}

#model_mlp = train_and_testMLP(data)
#print(model_mlp.layers[1].get_weights())

#model_cnn = train_and_testCNN(data)
#print(model_cnn.layers[2].get_weights())

#model_lstm = train_and_testLSTM(data)
#print(model_lstm.layers[1].get_weights())

#benchmark(data)




