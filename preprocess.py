#%% 
import pandas as pd
import os
import numpy as np
#%% ------path
data_path = '/home/j/jifang/CS5228-knowledge-discovery-and-data-mining/data'
data_path_train = os.path.join(data_path,'train/train')
file = os.path.join(data_path,'train_kaggle.csv')
pro_file = os.path.join(data_path,'train_kaggle_pro.csv')
data_path_test = os.path.join(data_path,'test/test')

df = pd.read_csv(file)
df.head()

#%%
doc = open(pro_file,'a',encoding='utf-8')
string = ''
for i in range(1,41):
    string = string + str(i)+','
string = string + 'label'
doc.write(string)
for i in range(0,len(df)):
    item = df.iloc[i]
    Id, label = item['Id'], item['label']
    feature = np.load(os.path.join(data_path_train,str(Id)+'.npy'))
    feature = np.nanmean(feature,axis=0)
    print(feature.shape)
    string = '\n'
    for i in feature:
        string = string + str(i)+','
    string = string + str(label)
    doc.write(string)
    print('----write no.%d record' %(Id))
doc.close()

#%%
data_pro = pd.read_csv(pro_file)
data_pro = data_pro.fillna(-1)
data_pro.head()
#%%
data_pro.iloc[0,0]

#%%
Y_train = data_pro.pop('label')
X_train = data_pro
print(Y_train)

#%%
from sklearn.model_selection import train_test_split

X_tra,X_val,y_tra,y_val = train_test_split(X_train,Y_train,test_size=0.25,random_state=33)

#%%
from sklearn.ensemble import RandomForestClassifier
import pickle


clf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1, class_weight="balanced")

clf.fit(X_train, Y_train)

with open('clf.pickle', 'wb') as fw:
    pickle.dump(clf, fw)

#%%
import glob
pro_file = os.path.join(data_path,'test_kaggle_pro.csv')
doc = open(pro_file,'a',encoding='utf-8')
string = ''
for i in range(1,41):
    string = string + str(i)+','
string = string + 'Id'
doc.write(string)
for i in range(0,10000):
    name = data_path_test+'/'+str(i)+'.npy'
    feature = np.load(name)
    string = '\n'
    feature = np.nanmean(feature,axis=0)
    for i in feature:
        string = string + str(i)+','
    # print(string)
    _,fename=os.path.split(name)
    fename = os.path.splitext(fename)
    string = string + str(fename[0])
    print(fename[0])
    doc.write(string)
doc.close()
#%%
pro_file = os.path.join(data_path,'test_kaggle_pro.csv')
data_pro = pd.read_csv(pro_file)
data_pro = data_pro.fillna(-1)
data_pro.head()

#%%
Y_test = data_pro.pop('Id')
X_test = data_pro

#%%
X_test.head()

#%%
y_pred = clf.predict_proba(X_test)

#%%
print(y_pred[:,1])

#%%
doc = open('submiss.csv','a',encoding='utf-8')
doc.write('Id,Predicted')
for i in range(0,len(y_pred)):
    string = '\n'
    string = string + str(i)+','+str(y_pred[i,1])
    doc.write(string)
doc.close()

#%%
