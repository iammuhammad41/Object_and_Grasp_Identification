import os
import csv
import re
import glob
import pandas as pd
# balancing the data
from imblearn.over_sampling import RandomOverSampler
#label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

root = '/media/song/新加卷/Action_Recognition/data'

label=[]
value=[]
for file in glob.glob(root + '/**/*.txt', recursive=True):
    # print(file)
    f = open(file, "r")
    if file.split('/')[-1] == 'grasp_class.txt':
        for i in f.read().split('\n'):
            if i.split(':') != ['']:
                label.append(i.split(':'))
    elif file.split('/')[-1] == 'skeleton.txt':
        for i in f.read().split('\n'):
            if i != '':
                value.append(i.split())



df=pd.DataFrame(value)
# df.dropna(inplace=True)
# print(df)

df1=pd.DataFrame(label)

# df1.dropna(inplace=True)

# print(df1)

df[0]=df1[1]

# print(df)
# print('nan values', df.isna().sum())
# print('nan values', df.isna())
df.rename(columns = {0:'Label'}, inplace = True)

# print(df)

df.dropna()

X=df.drop(['Label'],axis=1)
y=df[['Label']]



y.Label=y.Label.str.strip()

y.Label.value_counts()
print('Count Labelled Classes', y.Label.value_counts())

# # balancing the data
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# #label encoding
lb = LabelEncoder()
y = lb.fit_transform(y)

set(y)
print('Labelled Classes: ', set(y))

dic={}
for i in range(len(set(y))):
    dic[list(set(y))[i]]=list(set(lb.inverse_transform(y)))[i]
#     print(list(set(y))[i])0
#     dic[lb.inverse_tranform(y)[i]]=y
print(len(dic))
print('Labelled Classes Dic: ', dic)

# from sklearn.preprocessing import MinMaxScaler
# transformer = MinMaxScaler()
# transformer.fit(X)
# X=transformer.transform(X)

'''Data split'''
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Val. Accuracy:  0.9479591648359763
Labelled Classes Dic:  {0: '31', 1: '0', 2: '8', 3: '18', 4: '6', 5: '3', 6: '27', 7: '15', 8: '9', 9: '24', 10: '7', 11: '13', 12: '28', 13: '4', 14: '16', 15: '11', 16: '23', 17: '17', 18: '12', 19: '2', 20: '26', 21: '20', 22: '33', 23: '14', 24: '19', 25: '22', 26: '29', 27: '1'}
F1 score: 0.947364767740435
'''