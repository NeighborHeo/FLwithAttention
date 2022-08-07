# %%
import argparse
import json
import os
import threading
import time
from random import random
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics, regularizers
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy_encoder
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import pickle
from tensorflow.keras.utils import to_categorical

# tf.random.set_random_seed(42)  # tensorflow seed fixing
# %%
'''
    - 데이터 로딩하기
'''


# global edge
# functions = {-1: load_central,
#              0: load_edge,
#              1: load_edge,
#              2: load_edge,
#              3: load_edge}

# func = functions[edge]
# func(edge)

# client_name = "edge_{}".format(edge)

def load_config():
    with open('./{}'.format("data_config.json"), encoding='UTF8') as file:
        cfg = json.load(file)
    return cfg
        
cfg = load_config()
drop_cols = cfg['drop_cols']
selected_cols = cfg['selected_cols']
target_col = cfg['target_col']
icu_units = list(cfg['icu_units'].values())

# In[]:
# file_path = parent_dir.joinpath('data', 'data_drop_outlier_df.feather')
# data_df = pd.read_feather(file_path)

# for icu in cfg['label_encoder']:
#     n_icu = cfg['label_encoder'][icu]
#     print(icu, n_icu)
#     file_path = parent_dir.joinpath('data', 'eicu')
#     pathlib.Path.mkdir(file_path, mode=0o777, parents=True, exist_ok=True)
#     icu_df = data_df[data_df['icu']==n_icu].reset_index(drop=True)
#     icu_df = icu_df[selected_cols + target_col]
#     icu_df.to_feather(file_path.joinpath(f'{icu}.feather'))


def _load_dataset_central():
    train_data_df = pd.DataFrame()
    for icu in icu_units[:-1]:
        icu_df = pd.read_feather(parent_dir.joinpath('data', 'eicu', f'{icu}.feather'))
        train_data_df = pd.concat([train_data_df, icu_df], axis=0).reset_index(drop=True)
    valid_data_df = pd.read_feather(parent_dir.joinpath('data', 'eicu', f'{icu_units[-1]}.feather'))
    return train_data_df, valid_data_df

def _load_dataset_edge(n):
    train_data_df = pd.read_feather(parent_dir.joinpath('data', 'eicu', f'{icu_units[n]}.feather'))
    valid_data_df = pd.read_feather(parent_dir.joinpath('data', 'eicu', f'{icu_units[-1]}.feather'))
    return train_data_df, valid_data_df

def init_client_name(n):
    global client_name
    if (n==-1):
        client_name = "central"
    else :
        client_name = "edge_{}".format(n)

def initPath(client_name):
    global current_dir, parent_dir, result_dir
    import pathlib
    current_dir = pathlib.Path.cwd()
    parent_dir = current_dir.parent
    result_dir = pathlib.Path.joinpath(parent_dir, 'result', 'eicu', client_name)
    pathlib.Path.mkdir(result_dir, mode=0o777, parents=True, exist_ok=True)

def load_dataset(n):
    if (n==-1):
        train_data_df, valid_data_df = _load_dataset_central()
    else :
        train_data_df, valid_data_df = _load_dataset_edge(n)

    X_data, y_data = train_data_df[selected_cols], train_data_df['death']
    X_valid, y_valid = valid_data_df[selected_cols], valid_data_df['death']
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, stratify=y_data, random_state=0)
    
    return X_train, X_test, X_valid, y_train, y_test, y_valid

# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)

# dataset = {"X_train":X_train, "y_train":y_train, 'X_valid':X_valid, 'y_valid':y_valid,'X_test':X_test, 'y_test':y_test}

# with open('./dataset.pkl','wb') as f:
#         pickle.dump(dataset, f)

# common만 선정
# X_train, X_valid, X_test = X_train[feature_book['common_features']], X_valid[feature_book['common_features']], X_test[feature_book['common_features']]
# y_train, y_valid, y_test = to_categorical(y_train), to_categorical(y_valid), to_categorical(y_test)

#%%


#%%

'''
    build ann model
'''
def build_nn_model(
        input_size=len(selected_cols), n_layers=4, n_hidden_units=[64,128,64,32],
        random_seed=None, num_classes=2
):
    """
        creates the MLP network
        :return: model: models.Model`
        """
    # create input layer
    input_layer = layers.Input(shape=input_size, name="input")
    # create intermediate layer
    dense = input_layer
    for i in range(n_layers):
        dense = layers.Dense(
            units=n_hidden_units[i],
            kernel_initializer=initializers.glorot_uniform(seed=random_seed),
            bias_initializer='zeros',
            activation='relu',
            name='intermediate_dense_{}'.format(i + 1)
        )(dense)
    output_layer = layers.Dense(num_classes,
                                kernel_initializer=initializers.glorot_uniform(seed=random_seed),
                                bias_initializer='zeros',
                                activation='softmax',
                                name='classifier')(dense)
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=['accuracy'])
    return model

# %%
'''
    request global_weight from server
'''
def request_global_weight():
    print("request_global_weight start")
    result = requests.get(ip_address)
    result_data = result.json()
    global_weight = None
    if result_data is not None:
        global_weight = []
        for i in range(len(result_data)):
            temp = np.array(result_data[i], dtype=np.float32)
            global_weight.append(temp)
    print("request_global_weight end")
    return global_weight

# %%
'''
    update local weight to server
'''
def update_local_weight(local_weight = []):
    print("update local weight start ")
    local_weight_to_json = json.dumps(local_weight, cls=numpy_encoder.NumpyEncoder)
    requests.put(ip_address, data=local_weight_to_json)
    print("update local weight end")
    
# %%
def getClassWeight(y_data):
    from sklearn.utils import class_weight
    weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                            classes = np.unique(y_data),
                                            y = y_data)
    weights = {i : weights[i] for i in range(2)}
    return weights

def train_local(global_weight = None):
    print("train local start")
    model = build_nn_model()
    global X_train
    global y_train
    if global_weight is not None:
        global_weight = np.array(global_weight)
        model.set_weights(global_weight)
    
    weights = getClassWeight(y_test)
    model.fit(X_train, y_train, epochs=200, batch_size=128, class_weight=weights)
    
    print("train local end")
    return model.get_weights()

# %%
def delay_compare_weight():
    print("current_round : {}, max_round : {}".format(current_round, max_round))
    if current_round < max_round:
        threading.Timer(delay_time, task).start()
    else:
        '''
        if input_number == 0:
            print_result()
        '''
# %%
def request_current_round():
    result = requests.get(request_round)
    result_data = result.json()
    return result_data

#%%
def request_total_round():
    result = requests.get(request_total_round_url)
    print("this is the total round : ",result.json())
    result_data =  result.json()
    return result_data

# %%
def validation(global_lound = 0, local_weight = []):
    print("validation start")
    if local_weight is not None:
        model = build_nn_model()
        model.set_weights(local_weight)
        y_pred = model.predict(X_valid)
        
        answer_vec = to_categorical(y_valid)
        answer = y_valid
        
        auroc_ovr = metrics.roc_auc_score(answer_vec, y_pred, multi_class='ovr')
        auroc_ovo = metrics.roc_auc_score(answer_vec, y_pred, multi_class='ovo')
        y_pred = np.argmax(y_pred, axis=1)
        performance_result = model_performance(y_valid, y_pred, y_pred)
        save_model_performance(performance_result)
        # save_result(model, global_lound, global_acc=acc, f1_score=f3, auroc=auroc_ovo)
        shapley_value_result = get_shapley_value(model)
        save_shapley_value(shapley_value_result)
        print("validation end")
# %%
def save_result(model, global_rounds, global_acc, f1_score, auroc):
    test_name=client_name
    create_directory("{}".format(test_name))
    create_directory("{}/model".format(test_name))
    if global_acc >= 0.8 :
        file_time = time.strftime("%Y%m%d-%H%M%S")
        model.save_weights("{}/model/{}-{}-{}-{}.h5".format(test_name, file_time, global_rounds, global_acc, f1_score))
    save_csv(test_name=test_name, round = global_rounds, acc = global_acc, f1_score=f1_score, auroc=auroc)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_csv(test_name = "", round = 0, acc = 0.0, f1_score = 0, auroc = 0):
    with open("{}/result.csv".format(test_name), "a+") as f:
        f.write("{}, {}, {}, {}\n".format(round, acc, f1_score, auroc))

def get_shapley_value(model):
    import shap
    from shap import sample
    # explain predictions of the model on four images
    e = shap.explainers.Permutation(model.predict, sample(X_train, 100))
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e(X_train[:100])

    # get just the explanations for the positive class
    shap_values = shap_values[...,1]
    shap.plots.bar(shap_values)
    vals = np.abs(shap_values.values).mean(0)
    feature_names = X_train.columns
    result = dict(zip(feature_names, vals))
    return result

def save_shapley_value(result):
    file_path = result_dir.joinpath('feature_importance.feather')
    result_df = pd.DataFrame()
    if file_path.exists() :
        result_df = pd.read_feather(file_path)
    add_result_df = pd.Series(result).to_frame(name='{}'.format(current_round)).transpose()
    result_df = pd.concat([result_df, add_result_df], axis=0).reset_index(drop=True)
    result_df.to_feather(file_path)
# %%
def task():
    print("--------------------")
    '''
        1. global weight request
        2. global weight & local weight compare
        3. start learning & validation processing            
        5. global weight update
        6. delay, next round
    '''
    global current_round
    global_round = request_current_round()
    print("global round : {}, local round :{}".format(global_round, current_round))
    
    total_round = request_total_round()
    
    if global_round == total_round :
        global_weight = request_global_weight()
        with open('final_global_weight.pkl', 'wb') as f:
            pickle.dump(global_weight,f)
        sys.exit(0)
        
    if global_round == current_round:
        print("task train")
        # start next round
        global_weight = request_global_weight()
        local_weight = train_local(global_weight)
        # validation 0 clinet
        if input_number == 0 :
            validation(global_round, global_weight)
        update_local_weight(local_weight)
        delay_compare_weight()
        current_round += 1
    else:
        print("task retry")
        delay_compare_weight()
    print("end task")
    print("====================")
    

# %%
def print_result():
    print("====================")

# %%
from tensorflow.python.keras.callbacks import EarlyStopping
def single_train():
    early_stopping = EarlyStopping(patience=5)
    model = build_nn_model()
    weights = getClassWeight(y_train)
    model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_data=[X_test, y_test], callbacks=[early_stopping], class_weight=weights)
    y_pred = model.predict(X_valid)
    # answer_vec = to_categorical(y_test)
    # auroc_ovr = metrics.roc_auc_score(answer_vec, y_pred, multi_class='ovr')
    # auroc_ovo = metrics.roc_auc_score(answer_vec, y_pred, multi_class='ovo')
    y_pred = np.argmax(y_pred, axis=1)
    # auroc = metrics.roc_auc_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # acc = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average=None)
    # f2 = f1_score(y_test, y_pred, average='micro')
    # f3 = f1_score(y_test, y_pred, average='macro')
    # f4 = f1_score(y_test, y_pred, average='weighted')
    # print("acc : {}".format(acc))
    # # print("auroc ovr : {}".format(auroc_ovr))
    # # print("auroc ovo : {}".format(auroc_ovo))
    # print("auroc : {}".format(auroc))
    # print("f1 : {}".format(f1))
    # print("f2 : {}".format(f2))
    # print("f3 : {}".format(f3))
    # print("f4 : {}".format(f4))
    # print("cm : ", cm)
    performance = model_performance(y_valid, y_pred, y_pred)
    save_model_performance(performance)
    shapley_value_result = get_shapley_value(model)
    save_shapley_value(shapley_value_result)
    
def save_model_performance(result):
    file_path = result_dir.joinpath('model_performance.feather')
    result_df = pd.DataFrame()
    if file_path.exists() :
        result_df = pd.read_feather(file_path)
    add_result_df = pd.Series(result).to_frame(name='{}'.format(current_round)).transpose()
    result_df = pd.concat([result_df, add_result_df], axis=0).reset_index(drop=True)
    result_df.to_feather(file_path)

def model_performance(y_true, y_pred, y_proba):
    from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    
    roc_auc = roc_auc_score(y_true, y_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    TP, FP, FN, TN = cm[0][0], cm[1][0], cm[0][1], cm[1][1]
    
    result = {}
    result['TP'] = TP
    result['FP'] = FP
    result['FN'] = FN
    result['TN'] = TN
    result['f1'] = f1_score(y_true, y_pred, average=None)
    result['f2'] = f1_score(y_true, y_pred, average='micro')
    result['f3'] = f1_score(y_true, y_pred, average='macro')
    result['f4'] = f1_score(y_true, y_pred, average='weighted')
    result['precision'] = TP/(TP+FP)
    result['specificity'] = TN/(TN+FP)
    result['sensitivity'] = TP/(TP+FN) 
    result['recall'] = result['sensitivity'] # recall = sensitivity
    result['accuracy'] = (TP+TN) / (FP+FN+TP+TN)
    result['f1score'] = 2*result['precision']*result['recall']/(result['precision']+result['recall'])
    result['roc_auc'] = roc_auc
    result['mcc'] = mcc
    
    print(result)
    return result
    out = open('{}/model_performance_evaluation.txt'.format(output_path),'a')
    out.write(str(outcome_name) + '///')
    out.write(str(TP) + '///')
    out.write(str(TN) + '///')
    out.write(str(FP) + '///')
    out.write(str(FN) + '///')
    out.write('{:.3}'.format(result['precision']) + '///')
    out.write('{:.3}'.format(result['specificity']) + '///')
    out.write('{:.3}'.format(result['accuracy']) + '///')
    out.write('{:.3}'.format(result['recall'])+ '///')
    out.write('{:.3}'.format(result['f1score'])+ '///')
    out.write('{:.3}'.format(result['roc_auc'])+ '///')
    out.write('{:.3}'.format(result['mcc']))
    out.write('\n')        
    out.close()
    
    print("TP", TP) 
    print("TN", TN)
    print("FP", FP)
    print("FN", FN)
    print("precision", '{:.3%}'.format(result['precision']))
    print("specificity", '{:.3%}'.format(result['specificity']))
    print("accuracy", '{:.3%}'.format(result['accuracy']))
    print("recall", '{:.3%}'.format(result['recall']))
    print("f1_score", '{:.3%}'.format(result['f1score']))    
    print("roc_auc", '{:.3%}'.format(result['roc_auc']))    
    print("mcc", '{:.3%}'.format(result['mcc']))    

    
# %%
if __name__ == "__main__":
    parameter = argparse.ArgumentParser()
    parameter.add_argument("--number", default=0)
    parameter.add_argument("--currentround", default=0)
    parameter.add_argument("--maxround", default=3000)
    parameter.add_argument("--local", default=False)
    parameter.add_argument("--edge", default=-1)
    args = parameter.parse_args()
    
    input_number = int(args.number)
    current_round = int(args.currentround)
    max_round = int(args.maxround)
    bLocal = bool(args.local)
    edge = int(args.edge)
    
    
    init_client_name(edge)
    initPath(client_name)
    X_train, X_test, X_valid, y_train, y_test, y_valid = load_dataset(edge)
    
    if bLocal:
        single_train()

    else :
        np.random.seed(42)
        np.random.seed(input_number)
        
        print("args : {}".format(input_number))
        global_round = 0
        delay_time = 5  # round check every 5 sec
        # split_train_images, split_train_labels = split_data(input_number)
        
        base_url = "http://127.0.0.1:8000/"
        ip_address = "http://127.0.0.1:8000/weight"
        request_round = "http://127.0.0.1:8000/round"
        request_total_round_url = "http://127.0.0.1:8000/total_round"
        
        start_time = time.time()
        task()