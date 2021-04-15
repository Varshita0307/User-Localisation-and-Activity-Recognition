import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob
import math
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import metrics
from keras import regularizers
from keras.layers import Dropout
from keras.constraints import maxnorm


train_val_set = pd.read_csv("/home/varshi/Desktop/Project/TrainingData.csv")
test_set = pd.read_csv("/home/varshi/Desktop/Project/ValidationData.csv")

train_val_set.iloc[:, 0:520].min().min() # minimum WAP is -104 dBm
train_val_set_P = train_val_set.copy()
train_val_set_P.iloc[:, 0:520] = np.where(train_val_set_P.iloc[:, 0:520] <= 0, 
                train_val_set_P.iloc[:, 0:520] + 105, 
                train_val_set_P.iloc[:, 0:520] - 100) 

train_val_set_PN = train_val_set_P.copy()
train_val_set_PN.iloc[:, 0:520].max().max()
train_val_set_PN.iloc[:, 0:520] = train_val_set_P.iloc[:, 0:520]/105

combined = pd.concat([train_val_set_PN, test_set]) # stack vertically
combined = combined.assign(UNIQUELOCATION = (combined['LONGITUDE'].astype(str) + '_' + combined['LATITUDE'].astype(str) + '_' + combined['FLOOR'].astype(str) + '_' + combined['BUILDINGID'].astype(str)).astype('category').cat.codes)
len(combined["UNIQUELOCATION"].unique()) # 1995 unique locations

train_val_set_PUN = combined.iloc[0:19937, :]
test_set_U = combined.iloc[19937:21048, :]

# Change variable types
train_val_set_PUN["UNIQUELOCATION"] = train_val_set_PUN["UNIQUELOCATION"].astype("category")
train_val_set_PUN.dtypes

# Since UNIQUELOCATION is a multi-class label... 
dummy = keras.utils.to_categorical(train_val_set_PUN['UNIQUELOCATION'], num_classes = 1995)
dummy = pd.DataFrame(dummy, dtype = 'int')
train_val_set_PUND = pd.concat([train_val_set_PUN, dummy] ,axis = 1)


X_train_val = train_val_set_PUND.iloc[:, 0:520]
y_train_val = train_val_set_PUND.iloc[:, 520:2525]

test_set_PU = test_set_U.copy()
test_set_PU.iloc[:, 0:520] = np.where(test_set_PU.iloc[:, 0:520] <= 0, test_set_PU.iloc[:, 0:520] + 105, test_set_PU.iloc[:, 0:520] - 100) 

# Feature Scaling - do not center - destroys sparse structure of this data. 
# Normalize the WAPs by dividing by 105. Speeds up gradient descent.
test_set_PUN = test_set_PU.copy()
test_set_PUN.iloc[:, 0:520].max().max()
test_set_PUN.iloc[:, 0:520] = test_set_PU.iloc[:, 0:520]/105

# Change variable types
test_set_PUN["UNIQUELOCATION"] = test_set_PUN["UNIQUELOCATION"].astype("category")
test_set_PUN.dtypes


# Since UNIQUELOCATION is a multi-class label... 
dummy = keras.utils.to_categorical(test_set_PUN['UNIQUELOCATION'], num_classes = 1995)
dummy = pd.DataFrame(dummy, dtype = 'int')
test_set_PUND = pd.concat([test_set_PUN, dummy] ,axis = 1)


X_test = test_set_PUND.iloc[:, 0:520]
y_test = test_set_PUND.iloc[:, 520:2525]

#Split processed train_val set into training set and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                                  test_size = 0.2, 
                                                  random_state = 0)


#Create a reference table for looking up the LONGITUDE, LATITUDE, FLOOR, and 
# BUILDINGID associated with each UNIQUELOCATION value.
ref_table = pd.concat([y_train.iloc[:, [0,1,2,3,9]], 
                       y_val.iloc[:, [0,1,2,3,9]],
                        y_test.iloc[:, [0,1,2,3,9]]])
ref_table = ref_table.drop_duplicates()

def elaborate(test,model,ui):
    ans = []
    y_pred = model.predict(test)
    y_pred = np.argmax(y_pred,axis=1)

    dict_loc = {}
    m_total = ref_table.shape[0]
    for i in range(m_total):
        key = int(ref_table.iloc[i]['UNIQUELOCATION'])
        value = ref_table.iloc[i, 0:4].values
        dict_loc[key] = value

    y_pred_pos = np.asarray([dict_loc[i] for i in y_pred])[:, 0:2] 
    y_pred_floor = np.asarray([dict_loc[i] for i in y_pred])[:, 2]
    y_pred_building = np.asarray([dict_loc[i] for i in y_pred])[:, 3]
    ans.append(y_pred_pos[ui])
    ans.append(y_pred_floor[ui])
    ans.append(y_pred_building[ui])
    return ans 

'''
classifier = load_model("loc.h5")

y_pred = np.argmax(classifier.predict(X_test[0], axis = 1))

y_test_pos = y_test.iloc[:, 0:2].values 
y_test_floor = y_test.iloc[:, 2].values
y_test_building = y_test.iloc[:, 3].values
#print("y_pred:",y_pred)

dict_loc = {}
m_total = ref_table.shape[0]
for i in range(m_total):
    key = int(ref_table.iloc[i]['UNIQUELOCATION'])
    value = ref_table.iloc[i, 0:4].values
    dict_loc[key] = value

y_pred_pos = np.asarray([dict_loc[i] for i in y_pred])[:, 0:2] 
y_pred_floor = np.asarray([dict_loc[i] for i in y_pred])[:, 2]
y_pred_building = np.asarray([dict_loc[i] for i in y_pred])[:, 3]
'''



