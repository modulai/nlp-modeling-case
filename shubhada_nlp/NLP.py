# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:26:24 2021

@author: edesshu
"""

import preprocessor as p
import pandas as pd



train_df = pd.read_csv('C:\\Doc\\twitter_dataset_full.csv')
print(train_df.count())
train_df = train_df.dropna()
train_df = train_df.drop_duplicates()
#train_df = train_df.head(1000)

def preprocess_tweet(row):
    text = row['message']
    #print(text)
    text = p.clean(text)
    return text

train_df['message'] = train_df.apply(preprocess_tweet, axis=1)

train_df.head()


####Remove special characters#############
train_df['message'] = train_df['message'].str.replace("[^a-zA-Z#]", " ")
#

####Remove stopwords###########
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
from nltk.tokenize import word_tokenize
tweetText=train_df['message']
train_df['message'] = tweetText.apply(word_tokenize)


train_df1 = pd.concat([train_df['message'], train_df['is_positive']], axis=1)
train_df1.head()

train_df['is_positive'] = pd.to_numeric(train_df['is_positive'])
#####Training##############

from sklearn.model_selection import train_test_split        
from collections import Counter
import pandas as pd
import numpy as np
import string
    


class TweetNBClassifier(object):

    def __init__(self, df_train):
        self.df_train = df_train
        self.df_pos = df_train.copy()[df_train['is_positive'] == 1]
        self.df_neg = df_train.copy()[df_train['is_positive'] == 0]
       

    def fit(self):
        
        Pr_pos = self.df_pos.shape[0]/self.df_train.shape[0]
        Pr_neg =self.df_neg.shape[0]/self.df_train.shape[0]
        
        self.Prior  = (Pr_pos, Pr_neg)

        self.pos_words = ','.join(map(str, self.df_pos['message']))
        self.neg_words = ','.join(map(str, self.df_neg['message']))
        
        
        
        all_words = ','.join(map(str, self.df_train['message']))
        
        self.vocab = len(Counter(all_words))

        wc_pos = len(','.join(map(str, self.df_pos['message'])))
        wc_neg = len(','.join(map(str, self.df_neg['message'])))
       
        self.word_count = (wc_pos, wc_neg)
        return self


    def predict(self, df_test):
        class_choice = [1,0]

        classification = []
        for tweet in df_test['message']:
            text = tweet#.split()

            val_pos = np.array([])
            val_neg = np.array([])
           
            for word in text:
                tmp_pos = np.log((self.pos_words.count(word)+1)/(self.word_count[0]+self.vocab))
                tmp_neg = np.log((self.neg_words.count(word)+1)/(self.word_count[1]+self.vocab))
                
                val_pos = np.append(val_pos, tmp_pos)
                val_neg = np.append(val_neg, tmp_neg)
                

            val_pos = np.log(self.Prior[0]) + np.sum(val_pos)
            val_neg = np.log(self.Prior[1]) + np.sum(val_neg)
            

            probability = (val_pos, val_neg)
            classification.append(class_choice[np.argmax(probability)])
        return classification


    def score(self, feature, target):

        compare = []
        for i in range(0,len(feature)):
            if feature[i] == target[i]:
                tmp ='correct'
                compare.append(tmp)
            else:
                tmp ='incorrect'
                compare.append(tmp)
        r = Counter(compare)
        accuracy = r['correct']/(r['correct']+r['incorrect'])
        return accuracy




X_train, X_test, y_train, y_test = train_test_split(train_df['message'], train_df['is_positive'], test_size=0.33, random_state=0)

df_train = pd.DataFrame()
df_test = pd.DataFrame()

df_train['message'] = X_train
df_train['is_positive'] = y_train
df_train = df_train.reset_index(drop=True)

df_test['message'] = X_test
df_test['is_positive'] = y_test
df_test = df_test.reset_index(drop=True)    

#########train and predict
tnb = TweetNBClassifier(df_train)
tnb = tnb.fit()
predict = tnb.predict(df_test)
print(predict)

#F1-Score and Accuracy#########

from sklearn.metrics import f1_score

df_test['is_positive'] = pd.to_numeric(df_test['is_positive'])
predict = pd.to_numeric(predict)

print("F1-Score",f1_score(df_test['is_positive'], predict, average='weighted'))

score = tnb.score(predict,df_test.is_positive.tolist())
print("Accuracy",score)

