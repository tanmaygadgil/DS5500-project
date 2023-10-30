import torch
import numpy as np
import pandas as pd
import os
import sys

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
from torch import optim

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm
from model import simple_model
import pdb

def predict(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    y_pred = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())

    return y_pred

if __name__ == "__main__":
    
    data = pd.read_csv('../../data/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.2/features_v2.csv')
    # data = pd.read_csv(data_path)
    
    # data.drop(columns=["Unnamed: 0"], inplace=True)
    feature_cols = [c for c in data.columns if "_" in c]
    sessions = list(range(1, 4))
    
    results = {}
    
    WINDOW = 1

    for i in tqdm(range(len(sessions)-WINDOW + 1)):
        
        # pdb.set_trace()
        # try:
        train_parts = sessions[:i] + sessions[i+WINDOW+1:]
        test_parts = sessions[i:i+WINDOW]
        print(test_parts)
        # X = data.loc[:, feature_cols].values
        # Y = (data.loc[:, 'gesture']-1).values
        train_df = data[data['session'].isin(train_parts)]
        test_df = data[data['session'].isin(test_parts)]
        print(len(test_df))
        x_train = train_df.loc[:, feature_cols].values
        y_train = (train_df.loc[:, 'gesture'] - 1).values

        x_test = test_df.loc[:, feature_cols].values
        y_test = (test_df.loc[:, 'gesture'] - 1).values
        scaler = StandardScaler()

        scaler.fit(np.vstack([x_train, x_test]))

        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        train_dataset = TensorDataset(torch.tensor(x_train).type(torch.float32), torch.tensor(y_train).type(torch.LongTensor))
        test_dataset = TensorDataset(torch.tensor(x_test).type(torch.float32), torch.tensor(y_test).type(torch.LongTensor))
        
        # Define batch size and whether to shuffle the data
        batch_size = 128
        shuffle = True

        # Create data loaders for training and testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = simple_model(x_train.shape[1], len(np.unique(y_train)))
        
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Define training parameters
        num_epochs = 100
        
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
            
            if (epoch + 1) % 20 == 0:

                print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
                
                
        y_test_pred = predict(model, test_loader)
        
        accuracy = accuracy_score(y_test, y_test_pred)
        precision_micro = precision_score(y_test, y_test_pred, average='micro')
        precision_macro = precision_score(y_test, y_test_pred, average='macro')
        recall_micro = recall_score(y_test, y_test_pred, average='micro')
        recall_macro = recall_score(y_test, y_test_pred, average='macro')
        f1_micro = f1_score(y_test, y_test_pred, average='micro')
        f1_macro = f1_score(y_test, y_test_pred, average='macro')
        
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision (Micro)': precision_micro,
            'Precision (Macro)': precision_macro,
            'Recall (Micro)': recall_micro,
            'Recall (Macro)': recall_macro,
            'F1 (Micro)': f1_micro,
            'F1 (Macro)': f1_macro
        }
        
        test_parts = [str(i) for i in test_parts]
        
        results[" ".join(test_parts)] = metrics_dict
        
        
    with open("session_test.json", "w") as file:
        json.dump(results, file)
        
        
        
                
                
                
        
            
        