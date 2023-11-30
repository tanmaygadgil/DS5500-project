import os
import sys
from scipy import stats

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

from model import simple_model, SimpleModel

NUM_EXP = 50
NUM_PART = 43
NUM_TRIALS = 2

def split_df_by_key(df, part_list, key="participant"):
    """Puts participants in a trainset

    Args:
        df (_type_): dataframe
        part_list (_type_): list of values in the train set
    """
    
    test_df = df[~df[key].isin(part_list)]
    train_df = df[df[key].isin(part_list)]
    
    return train_df, test_df

# def split_df_by_participants(df, participants, key="participant"):
#     """Puts participants in a testset

#     Args:
#         df (_type_): dataframe
#         participants (_type_): list of participants in the test set
#     """
    
#     train_df = df[~df[key].isin(participants)]
#     test_df = df[df[key].isin(participants)]
    
    return train_df, test_df
def extract_features(df:pd.DataFrame, feature_cols:list, scaler:StandardScaler = None ):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        feature_cols (list): _description_
        scaler (StandardScaler, optional): _description_. Defaults to None.
    """
    X = df.loc[:, feature_cols].values
    Y = (df.loc[:, 'gesture']-1).values
    if scaler:
        X = scaler.transform(X)
        return X, Y, scaler
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, Y, scaler
    
def train(model, train_loader, test_loader, num_epochs, optimizer, criterion):
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate and log metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(test_loader)
        val_accuracy = 100 * correct / total

        if (epoch+1)%20==0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - \
                Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, \
                    Val Accuracy: {val_accuracy:.2f}%")
        
    return model    

def predict(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    y_pred = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())

    return y_pred

def get_results(y_test, y_part_pred):
    accuracy = accuracy_score(y_test, y_part_pred)
    precision_micro = precision_score(y_test, y_part_pred, average='micro')
    precision_macro = precision_score(y_test, y_part_pred, average='macro')
    recall_micro = recall_score(y_test, y_part_pred, average='micro')
    recall_macro = recall_score(y_test, y_part_pred, average='macro')
    f1_micro = f1_score(y_test, y_part_pred, average='micro')
    f1_macro = f1_score(y_test, y_part_pred, average='macro')
    
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision (Micro)': precision_micro,
        'Precision (Macro)': precision_macro,
        'Recall (Micro)': recall_micro,
        'Recall (Macro)': recall_macro,
        'F1 (Micro)': f1_micro,
        'F1 (Macro)': f1_macro
    }
    
    
    return metrics_dict
            

    
if __name__ == "__main__":
    
    data = pd.read_csv('../../data/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.2/features_v2.csv')
    #removing extreme outliers
    feature_cols = [c for c in data.columns if "_" in c]
    data = data[(np.abs(stats.zscore(data.loc[:, feature_cols])) < 5.5).all(axis = 1)].reset_index()
    
    untuned_model_results = {}
    tuned_model_results = {}
    #for loop to conduct N Exxperiments
    for i in tqdm(range(NUM_EXP)):
        #get 5 random participants
        # print(f"Experiment number: {i+1}")
        test_participants = np.random.choice(range(1,44), 5, False)
        # test_participants = [2,4,13,25,39]
        train_participants = list(set(range(1,44)) - set(test_participants))
        # print(f"analyzing participants: {test_participants}")
        ##extract training and test df
        train_df, test_df = split_df_by_key(data, train_participants, "participant")
        # print(f"Participant:{test_df.participant.unique()}")
        #Get train features
        x, y, scaler = extract_features(train_df, feature_cols)
        x_test, y_test, scaler = extract_features(test_df, feature_cols, scaler)
        
        #training params
        train_dataset = TensorDataset(torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.LongTensor))
        test_dataset = TensorDataset(torch.tensor(x_test).type(torch.float32), torch.tensor(y_test).type(torch.LongTensor))
        
        # Define batch size and whether to shuffle the data
        batch_size = 256
        shuffle = True

        # Create data loaders for training and testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = simple_model(x.shape[1], len(np.unique(y)))
        # model = SimpleModel(x.shape[1], len(np.unique(y)))
        
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        
        num_epochs = 60
        
        #train model
        model = train(model, train_loader, test_loader, num_epochs, optimizer, criterion)
        
        
        
        test_parts_str = " ".join([str(i) for i in test_participants])
        
        untuned_model_results[test_parts_str] = {}
        tuned_model_results[test_parts_str] = {}
        
        #test for each participant
        for idx, part in enumerate(test_participants):
            # print(f"analyzing test participant: {part}")
            part_df = test_df[test_df['participant'] == part]
            x_part, y_part, scaler = extract_features(part_df, feature_cols, scaler)
            test_dataset = TensorDataset(torch.tensor(x_part).type(torch.float32), torch.tensor(y_part).type(torch.LongTensor))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            #predict performance
            y_part_pred = predict(model, test_loader)
            metrics_dict = get_results(y_part, y_part_pred)
        
            untuned_model_results[test_parts_str][int(part)] = metrics_dict
        
        
        #loop over all participants for tuning
        for idx, part in enumerate(test_participants):
            # print(f"tuning participant: {part}")
            # new_model = model.clone()
            new_model = simple_model(x.shape[1], len(np.unique(y)))
            new_model.load_state_dict(copy.deepcopy(model.state_dict())) 
            # new_model = copy.deepcopy(model)
            print(type(new_model))
            trials = range(1, NUM_TRIALS+1)
            part_df = test_df[test_df['participant'] == part]
            tune_train_df, tune_test_df = split_df_by_key(part_df, trials, key = "trial")
            
            print(f"Participant:{tune_train_df.participant.unique()} - unique trials {tune_train_df.trial.unique()}")
            print(f"Participant:{tune_test_df.participant.unique()} - unique trials {tune_test_df.trial.unique()}")
            
            x_tune, y_tune, scaler = extract_features(tune_train_df, feature_cols, scaler)
            x_tune_test, y_tune_test, scaler = extract_features(tune_test_df, feature_cols, scaler)
            
            #training params
            tune_train_dataset = TensorDataset(torch.tensor(x_tune).type(torch.float32), torch.tensor(y_tune).type(torch.LongTensor))
            tune_test_dataset = TensorDataset(torch.tensor(x_tune_test).type(torch.float32), torch.tensor(y_tune_test).type(torch.LongTensor))
            
            # Define batch size and whether to shuffle the data
            batch_size = 256
            shuffle = True

            # Create data loaders for training and testing
            tune_train_loader = DataLoader(tune_train_dataset, batch_size=batch_size, shuffle=shuffle)
            tune_test_loader = DataLoader(tune_test_dataset, batch_size=batch_size, shuffle=False)
            
            
            criterion = nn.CrossEntropyLoss()  
            optimizer = optim.Adam(new_model.parameters(), lr=5e-4)
        
            num_epochs = 20
            
            new_model = train(new_model, tune_train_loader, tune_test_loader, num_epochs, optimizer, criterion)
            
            y_tune_pred = predict(new_model, tune_test_loader)
            
            metrics_dict = get_results(y_tune_test, y_tune_pred)
            
            
            tuned_model_results[test_parts_str][int(part)] = metrics_dict
            
            
    with open(f"untuned_model_results_{NUM_TRIALS}_trials.json", 'w') as file:
        json.dump(untuned_model_results, file)
        
    with open(f"tuned_model_results_{NUM_TRIALS}_trials.json", 'w') as file:
        json.dump(tuned_model_results, file)