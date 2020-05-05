from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.layers import Conv2D, LSTM, Conv1D, MaxPooling2D
from tensorflow.keras.layers import Input, Activation, RepeatVector, Reshape
from tensorflow.keras.layers import multiply, Permute, GlobalAveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score 
from keras.callbacks.callbacks import EarlyStopping
import numpy as np
from matplotlib import pyplot as plt
import copy

def print_result(name, Y_true, Y_pred):
    p = precision_score(Y_true, Y_pred,average='macro') 
    r = recall_score(Y_true, Y_pred,average='macro')
    f1 = f1_score(Y_true, Y_pred,average='macro')
    kappa = cohen_kappa_score(Y_true, Y_pred)
    print('\nResults for', name, p, r, f1, kappa)
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

def benchmark(data, test_lenght):  
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

def split_train_test(X, Y, start_train, train_lenght, test_lenght):
    # Split formatted data into train and test. First train_period days for
    # training and the rest for testing
    # Exceptions?
    
    if (start_train + train_lenght + test_lenght) > X.shape[0]/N_TOP_ACTIVE:
        print('Check numbers')
        return
    
    N = N_TOP_ACTIVE
    
    X_train = X[start_train*N:(train_lenght + start_train)*N,:]  
    Y_train = Y[start_train*N:(train_lenght + start_train)*N,:]
    
    start_test = start_train + train_lenght
    
    X_test = X[start_test*N:(test_lenght + start_test)*N,:]    
    Y_test = Y[start_test*N:(test_lenght + start_test)*N,:]  
    
    return X_train, X_test, Y_train, Y_test

def format_data(data, days):
    # Format data to sequences of lenght days
    
    features = 13
    
    X = np.zeros([(data.shape[0] -days)*N_TOP_ACTIVE, days, features])    
    Y = np.zeros([(data.shape[0] -days)*N_TOP_ACTIVE, 1])
    
    pos = 0
    for j in range(data.shape[0] - days):
        for i in range(N_TOP_ACTIVE):
            # Investor i activity, eurovolume and time since last trade
            matrix = data[j:days+j,(i, i+N_TOP_ACTIVE, i+2*N_TOP_ACTIVE)] 
            
            # Number of active investors and market data
            matrix = np.hstack([matrix, data[j:days+j, -10:]])
            
            X[pos] = copy.deepcopy(matrix)
            Y[pos] = data[j + days, i]
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

def mlp_model(input_shape, neurons, net_width, net_depth):
    
    input_height = input_shape[2]

    inputs = Input(shape=(input_height))
    layer = inputs
    
    for i in range(net_depth):
        layer = Dense(neurons*net_width, activation='softmax')(layer)
        
    outputs = Dense(num_classes, activation='softmax')(layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[]) #'accuracy', get_f1. Per class accuracy?
    return model

def lstm_model(input_shape, neurons, net_width, net_depth):
    
    time_steps = input_shape[1]
    features = input_shape[2]
    
    inputs = Input(shape=(time_steps, features))
    layer = inputs
        
    for i in range(net_depth):
        # return sequences True, except on the last layer
        return_seq = True if not (i+1 == net_depth) else False
            
        layer = LSTM(neurons*net_width, activation='softmax', 
                     return_sequences=return_seq)(layer)
    
    outputs = Dense(num_classes, activation='softmax', dynamic=True)(layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy', get_f1]) 
    return model

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
                
    outputs = Dense(num_classes, activation='softmax')(layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[]) #'accuracy', get_f1
    return model


def train_and_test_model(X, Y, model_type):
    
    neurons = 16 
    all_results = list()
    
    if model_type == 'MLP':
        widths = [1, 2, 3]
        depths = [1, 2, 3]
    else:
        widths = [1, 2, 3, 4]
        depths = [1, 2]

    for net_width in widths:
        for net_depth in depths:
            
                if model_type == 'MLP':
                    model = mlp_model(X.shape, neurons, net_width, net_depth)
                elif model_type == 'CNN':
                    model = cnn_model(X.shape, net_width, net_depth)
                elif model_type == 'LSTM':
                    model = lstm_model(X.shape, neurons, net_width, net_depth)
                    
                name = model_type + str(net_width) + '-' + str(net_depth)
                model_results = list() 
                
                total_trainings = 12
                start = 0
                test_lenght = 21
                train_lenght = 661 
                
                epochs = 10 #?
                
                train_set = 1 
                while train_set <= total_trainings:
                
                    X_train, X_test, Y_train, Y_test = split_train_test(
                        X, Y, start, train_lenght, test_lenght)
                    
                    if model_type == 'MLP':
                        X_train = X_train[:,-1,:]
                        X_test = X_test[:,-1,:]
                    # Shape for CNN?
                    
                    print(name)
                    print('Training set {} from {} to {}, with test until {}'\
                          .format(train_set, start, start + train_lenght, 
                                  start + train_lenght + test_lenght))
                        
                    model.fit(X_train, Y_train, epochs=epochs, verbose=1,
                              batch_size=32, class_weight=class_weights)
                    
                    Y_true = np.argmax(Y_test, axis=1)
                    prob = model.predict(X_test)
                    Y_pred = np.argmax(prob, axis=1)
                    
                    p, r, f1, kappa = print_result(name, Y_true, Y_pred)
                    model_results.append([p, r, f1, kappa])
                    
                    train_set += 1
                    start += train_lenght
                    train_lenght = test_lenght
                    #epochs = 10 # should epochs change?
                
                print('\nMean results for', name, ':',
                      np.mean(np.asarray(model_results),axis=0),'\n\n')
                all_results.append([name, np.mean(np.asarray(model_results), 
                                             axis=0)])
                
    for row in all_results:
        print(row)
    
    return all_results

loc = 'C:\\Users\\enehl\\OneDrive - TUNI.fi\\Opiskelu\\Kandi'  
data = np.load(loc + '\\Code\\data.npy')

N_TOP_ACTIVE = 100
num_classes = 3
window = 20
class_weights = {0: 1.,
                1: 8.,
                2: 9.5}

X, Y = format_data(data, window)
Y = to_categorical(Y, num_classes)

# 21 day walk forward validation
results_together = list()
for model_type in ['MLP']:
    result = train_and_test_model(X, Y, model_type)
    
    for row in result:
        results_together.append(row)

print('All results together:')
for row in results_together:
    print(row)
    
#model_mlp = train_and_testMLP(data)
#print(model_mlp.layers[1].get_weights())

#model_cnn = train_and_testCNN(data)
#print(model_cnn.layers[2].get_weights())

#model_lstm = train_and_testLSTM(data)
#print(model_lstm.layers[1].get_weights())

#benchmark(data, 12*21)




