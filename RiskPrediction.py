# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 05:55:39 2024

@author: FCI1590
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import csv

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

# Tweaking Layers
# Dropout
# Missing Value Imputation
# Checking Clusters
# Tracing Losses
# Storing Metrics
# Google Colab
# Experimenting Layers
# Activation Functions
# Entire Data
# Parallel Execution

exec(open("TuneLD2.py").read())
exec(open("DatasetCreator.py").read())
exec(open("Parameters.py").read())


#exec(open("/content/gdrive/MyDrive/Defaulters/TuneLD2.py").read())
#exec(open("/content/gdrive/MyDrive/Defaulters/DatasetCreator.py").read())
#exec(open("/content/gdrive/MyDrive/Defaulters/Parameters.py").read())

#nn.Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from TuneLD2 import ProcessData
from DatasetCreator import TrainDataset,TestDataset,StratifiedBatchSampler
from Parameters import Arguments

class RiskModel(nn.Module):
    
    def __init__(self,no_features,kernel_features):
        
        super(RiskModel, self).__init__()
        
        hidden_node_size = no_features + kernel_features
        
        if self.layers == 1:
            self.VanillaRiskModel = nn.Sequential(
                nn.Linear(no_features, hidden_node_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_node_size, hidden_node_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_node_size, 1))
                
        elif self.layers == 2:
            self.VanillaRiskModel = nn.Sequential(
                nn.Linear(no_features, hidden_node_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_node_size, hidden_node_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_node_size, hidden_node_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_node_size, 1))
        
        
            #nn.Softmax(dim=1)
         
        
    def forward(self, x):
        
        return self.VanillaRiskModel(x)

class PredictionModel( ProcessData, Arguments ):

      
    def __init__(self, file_name,compute_flag, org_cluster, ocp_cluster,layers,kernel_size,dropout,batch_size,stratify,epochs,entire):

        
        super(ProcessData,self).__init__( file_name, compute_flag, org_cluster, ocp_cluster)
        super(Arguments,self).__init__( layers, kernel_size, dropout, batch_size, stratify, epochs, entire )
        
        self.create_data = self.create_data
        self.train_model = self.train_model
        self.predict_defaulters = self.predict_defaulters
        self.deal_categorical = self.deal_categorical
        self.run_model = self.run_model
        self.cat_label_indexer = self.cat_label_indexer
        self.bce_custom_criterion = self.bce_custom_criterion
        self.misssg_np_values = self.misssg_np_values

        
        self.pos_weight = 0.8
        self.neg_weight = 0.2

    def create_data(self):

        self.read_data(self.file_name)
        
        self.file_string = self.model_monitor()
        
        columns = self.column_types()
        cat_columns = columns["Categorical"]
        num_columns = columns["Numerical"]
        
        self.train.replace({"nan":np.nan},inplace= True)
        self.test.replace({"nan":np.nan},inplace = True)
       
        
        self.train = self.train.dropna(axis =0)
        self.test  = self.test.dropna(axis = 0)

        print(self.train.shape,self.test.shape)
        self.train_y = self.train["Default"]
        self.test_y = self.test["Default"]
        
        num_positives = self.train.loc[self.train["Default"] == "1" ].shape[0]
        num_negatives = self.train.shape[0] - num_positives
        
        self.pos_weight  = num_negatives / num_positives
        
        #self.pos_weight  = num_negatives / (num_negatives + num_positives)
        #self.neg_weight  = num_positives / num_negatives + num_positives
        
        print(self.train.Default.value_counts(),self.test.Default.value_counts())

        train_data = self.train.drop(columns="Default",axis = 1)
        test_data = self.test.drop(columns="Default",axis = 1)
        
        cat_columns.remove("Default")
        self.train_data,self.test_data = self.deal_categorical(self.train_y,train_data[num_columns],train_data[cat_columns],test_data[num_columns],test_data[cat_columns])
        
        
        #self.train_data,self.test_data = self.train_data[num_columns],self.test_data[num_columns]
        
        #self.train_data.to_csv("Training_data.csv")
       
      
        #self.train_data = self.train_data.iloc[:,1:10]
        #self.test_data = self.test_data.iloc[:,1:10]
        
        #self.train_data = self.train_data[num_columns]
        #self.test_data = self.test_data[num_columns]
        
        self.train_data= np.array(self.train_data.values, dtype=np.float64 )
        self.test_data = np.array(self.test_data.values,dtype = np.float64 )
       
        self.train_data[np.isnan(self.train_data)] = 0
        self.test_data[np.isnan(self.test_data)] = 0
        
        #self.train_data = self.misssg_np_values(self.train_data)
        #self.test_data = self.misssg_np_values(self.test_data)
   
        
        #self.train_data = self.train_data[~(np.isnan(self.train_data)).any(axis =1),:]
        #self.test_data = self.test_data[~(np.isnan(self.test_data)).any(axis =1),:]
        
        #self.test_data = self.test[np.logical_not(np.isnan(self.test_data))]

        
        self.no_features = self.train_data.shape[1] 
        
        train_set = TrainDataset(torch.from_numpy(self.train_data))
        test_set  = TestDataset(torch.from_numpy(self.test_data))
        
        y = np.array(self.train_y.values, dtype=np.float64 )
        self.y  = torch.from_numpy(y)
       
        if stratify:
            self.train_loader = DataLoader(train_set, batch_sampler=StratifiedBatchSampler(y, batch_size=self.batch_size,shuffle= True))
        else:
            self.train_loader = DataLoader(train_set, shuffle=True,batch_size= self.batch_size)
            
        self.test_loader  = DataLoader(test_set,  batch_size= self.batch_size, shuffle=False)

        

    def misssg_np_values(self,a):
    
        col_mean = np.nanmean(a, axis=0)
        inds = np.where(np.isnan(a))
        
        #a[inds] = np.take(col_mean, inds[1])
        
        a = np.where( np.isnan(a),col_mean,a )
        
        return a
    
    def deal_categorical(self,y,train_numerical,train_categorical,test_numerical,test_categorical):

        train_category,test_category = self.cat_label_indexer(train_categorical,test_categorical)
        #train_data = pd.concat([ y.set_axis(train_numerical.index),train_numerical,train_category.set_axis(train_numerical.index)],axis=1)
        
        train_data = pd.concat([ train_numerical,train_category.set_axis(train_numerical.index)],axis=1)

        test_data = pd.concat([ test_numerical,test_category.set_axis(test_numerical.index) ],axis=1)
            
        return train_data,test_data

    def cat_label_indexer(self,train_data,test_data):
        
        #le = LabelEncoder()

        one_hot_encoded_training_predictors = pd.get_dummies( train_data,columns =train_data.columns, dtype=float )
        one_hot_encoded_test_predictors = pd.get_dummies( test_data,columns=test_data.columns, dtype=float )
        
        final_train, final_test = one_hot_encoded_training_predictors.align( one_hot_encoded_test_predictors, join='left', axis=1 )
        
        return final_train, final_test

    
    def train_model(self):
        
        self.model = RiskModel(self.no_features,self.kernel_size).to(device)
        #self.model = RiskModel(self.no_features,self.kernel_size)
        
        if False:
            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, 0.0, 1.0))
                optimizer = torch.optim.Adam(self.model.parameters())
            
            self.model.to(device)
        
        lr = 1e-5
        weight_decay = 1e-7
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) 
        
        optimizer = torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.4,weight_decay = weight_decay)
        
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
        #loss_function = nn.NLLLoss()

        
        pos_weight = torch.tensor([self.pos_weight])
        criterion = nn.BCEWithLogitsLoss(pos_weight)
        
        #criterion = nn.BCELoss()
        
        #weight = torch.tensor([self.pos_weight,self.neg_weight])
        
        #criterion = nn.CrossEntropyLoss(weight=weight)

        self.model.train()
           
        for epoch in range(self.epochs):
            losses = []
            
            start = 0
            for batch_num, input_data in enumerate(self.train_loader):
                optimizer.zero_grad()
                #x,y = input_data
                x = input_data
                x = x.to(device).float()

                y = self.y[start : start +len(x) ]
                y = y.unsqueeze(1)
                #y = y.to(device).type(torch.long)
                y = y.to(device).float()
                
                start = start +len(x)
                
                output = self.model(x)
               
                #loss = criterion(output, y)
                
                #loss = self.bce_custom_criterion(output, y)
                loss = criterion(output, y)
                loss.backward()
                losses.append(loss.item())
                
               
                # Gradient Clipping
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., norm_type=2)
                
                #scaler.step(optimizer)
                
                optimizer.step()

            #   if batch_num % 40 == 0:
                if True:
                    print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
                    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
                pass
        
        if False:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
                

    def predict_defaulters(self,actual):
        
        
        self.model.eval()
        
        with open(self.file_string+'.pkl','w') as fp:
        
            fp.write(self.model)
        
        test_out = []

        with open(self.file_string +'_' +'mlp_submission.csv', 'w') as f:
        
            fieldnames = [ 'Customer', 'Label','Actual' ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator = '\n')
            writer.writeheader()
            customer_id = 1

            with torch.no_grad():
                
                for x in self.test_loader:
                    x = x.to(device).float()

                    out = self.model(x)
                    out = torch.sigmoid(out)
                    test_out = test_out + out.tolist()
                    
                    for y in out:
                        writer.writerow({fieldnames[0]: customer_id,fieldnames[1]: y.item(), fieldnames[2] : actual[customer_id]})
                        customer_id += 1
                        
        test_out = sum(test_out,[])
        
        return test_out
        
    # Define the loss function (criterion)
    def bce_custom_criterion(self, y_pred, y_true):
        # Binary Cross Entropy Loss
        # y_pred: predicted probabilities, y_true: true labels (0 or 1)
    
        # Compute the negative log likelihood loss using binary cross-entropy formula
        # (y * log(y_pred) + (1 - y) * log(1 - y_pred))
        loss = -1 * (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
        # Calculate the mean loss over the batch
        mean_loss = torch.mean(loss)
    
        return mean_loss

    def run_model(self):

        self.train_model()
        y_pred = self.predict_defaulters(self.test_y)
        
        #print(y_pred)
        #print(type(y_pred))

        self.posterior_cutoff(self.test_y,y_pred)

#exec(open("RiskPrediction.py").read())
#pdt = PredictionModel("../Data/Dataset.csv",False,4,4,100000,5,10)
#pdt.create_data()
#pdt.run_model()
