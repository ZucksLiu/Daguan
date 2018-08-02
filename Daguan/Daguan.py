print("开始.......")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

train1 = pd.read_csv('C:/Users/38084/train_set.csv',chunksize=10000)
test1 = pd.read_csv('C:/Users/38084/test_set.csv',chunksize=10000)
df_train=pd.DataFrame(columns=('id','article', 'word_seg', 'class'))
df_test=pd.DataFrame(columns=('id','article','word_seg'))
for chunk in train1:
    if df_train.empty is True:

        df_train = chunk

    else:
        df_train = pd.concat([df_train,chunk])

    print(chunk)
for chunk in test1:
    if df_test.empty is True:

        df_test = chunk

    else:
        df_test = pd.concat([df_test,chunk])
    df_test= pd.concat([df_test,chunk])
    print(chunk)

print('finish 0')

df_train.drop(['article','id'],axis=1, inplace=True)
df_test.drop(['article'],axis=1, inplace=True)


vectorizer = TFCountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features= 100000)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
y_train = df_train['class']-1
x_test = vectorizer.transform(df_test['word_seg'])
print('finish 1')

lg = LogisticRegression(C=4, dual=True)
lg.fit(x_train,y_train)

y_test = lg.predict(x_test)

df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class']+1
df_result = df_test.loc[:, ['id','class']]
df_result.to_csv('D:/BaiduNetdiskDownload/new_data/result.csv',index=False)

print('Finish.........')