# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:41:00 2022

@author: anilhr
"""
import re
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from textblob import TextBlob
from flashtext import KeywordProcessor
from textaugment import EDA
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
import pickle

nlp = spacy.load("en_core_web_sm")
import numpy as np
df= pd.read_excel('1000.xlsx',names=['Lead','Location','Status','information'])
df=df.dropna(axis=0)
key_dic = {
    'call' : ['cal'],'tomarrow': ['tmrw'],'evening':['evng'],'demo':['domo'],'busy':['busi'],
    'try':['tri'],'and':['nd','n'],'readable':['readabl'],'morning':['morn'],'already':['alreadi'],
    'please':['pleas'],'intrest':['intrust','interestand,detail','intrustneed','intrustshare','inttstd'],'online':['onlin'],'or':['r'],'unable':['unabl'],
    'issue':['issu'],'the':['di'],'register':['regist'],'require':['requir'],'manage':['manag'],
    'enquiry':['inquir'],'possible':['possibl'],'demo share':['domoshare'],'another':['anoth'],
    'course':['cours'],'you':['u'],'experience':['experi'],'require':['requir'],'try':['tri'],
    'details':['detailswitch'],'share':['shareshare,'],'readable':['readabl'],'sorry':['sorri'],'readable':['readab']
}
key_word = KeywordProcessor()
key_word.add_keywords_from_dict(key_dic)
df=df.replace(to_replace="NOt Converted",value="Not Converted")
df=df.replace(to_replace="Not Converted",value=1)
df=df.replace(to_replace="Conveted",value=0)
df = df.replace(to_replace="Converted ",value=0)


def augment_text(df,samples):
    eda_augment = EDA()
    new_text=[]
    
    ##selecting the minority class samples
    df_n=df[df.Status==0].reset_index(drop=True)

    ## data augmentation loop
    for i in tqdm(np.random.randint(0,len(df_n),samples)):
        
            text = df_n.iloc[i]['information']
         
            augmented_text = eda_augment.random_insertion(text)
            
            new_text.append(augmented_text)
    
    
    ## dataframe
    new=pd.DataFrame({'information':new_text,'Status':0})
    #df=shuffle(df.append(new).reset_index(drop=True))
    return new_text
   
def dataProcessing(df):
    corpus=[]
    p = PorterStemmer()
    for i in df['information']:
        pattern = '(?<=:).*?(?=\d)|(?<=:).*'
        matches = re.findall(pattern, i)
        matches = " ".join(matches)
        matches = matches.lower()
        matches = TextBlob(matches)
        matches = matches.correct()
        matches = matches.split()
        next_text = [key_word.replace_keywords(word)for word in matches]
        matche = " ".join(next_text)
        matches = matche.split()


        matches = [p.stem(word) for word in matches if not word in stopwords.words('english')]
        matche = " ".join(matches)
        corpus.append(matche)
    return corpus
    
data=augment_text(df,samples=700)
new=pd.DataFrame({'information':data,'Status':0})
frames = [df, new]

df_leads = pd.concat(frames).reset_index(drop=True)
df_leads=df_leads[['Status','information']]
df_leads["contents_new"] = dataProcessing(df_leads)


y = df_leads.Status
x = df_leads.contents_new

cv = CountVectorizer(ngram_range = (1, 2), max_features=5000)
X = cv.fit_transform(x) # Fit the Data
pickle.dump(cv, open('tranform.pkl', 'wb'))


clf = MultinomialNB()
clf.fit(X,y)
print(clf.score(X,y))
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))