# Required packages
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import re,gc,time
from sklearn.linear_model import Ridge


train = pd.read_csv('../input/train.tsv',sep='\t',engine='c')
test = pd.read_csv('../input/test.tsv',sep='\t',engine='c')
print('Importing data completed')

og_trn_len = train.shape[0]
print('Original train dataset length:{}'.format(og_trn_len))

Less_price_df:pd.DataFrame = train.loc[train.price<3]
print('Less price(<$3) dataset length:{}'.format(len(Less_price_df)))

train.drop(Less_price_df.index,inplace=True)
del(Less_price_df['price'])

y_train = np.log1p(train.price)
Red_trn_len = train.shape[0]
print('Reduced train data length:{}'.format(Red_trn_len))

merged = pd.concat([train,Less_price_df,test],axis=0)

t_id = test.test_id
del train,test,Less_price_df
gc.collect()

def Na_remover(data):
    data.brand_name.fillna('unknown',inplace= True)
    data.item_description.fillna('unknown',inplace= True)
    data.loc[data.item_description== 'No description yet','item_description']= 'unknown'
    data.category_name.fillna('unknown',inplace= True)
    
def Category_splitter(txt):
    try: return txt.split('/')
    except: return ['unknown','unknown','unknown']

def Cutting(data):
    brand_select = data.brand_name.value_counts().loc[lambda x: x.index!='unknown'].index[:4000]
    data.loc[~data.brand_name.isin(brand_select),'brand_name'] = 'unknown'
    cat1_select = data.cat1.value_counts().loc[lambda x: x.index!='unknown'].index[:1000]
    data.loc[~data.cat1.isin(cat1_select),'cat1'] = 'unknown'
    cat2_select = data.cat2.value_counts().loc[lambda x: x.index!='unknown'].index[:1000]
    data.loc[~data.cat2.isin(cat2_select),'cat2'] = 'unknown'
    cat3_select = data.cat3.value_counts().loc[lambda x: x.index!='unknown'].index[:1000]
    data.loc[~data.cat3.isin(cat3_select),'cat3'] = 'unknown'
    
def Category_maker(data):
    data.item_condition_id= data.item_condition_id.astype('category')
    data.brand_name= data.brand_name.astype('category')
    data.shipping= data.shipping.astype('category')
    data.cat1= data.cat1.astype('category')
    data.cat2= data.cat2.astype('category')
    data.cat3= data.cat3.astype('category')
    
def desc_finder(txt):
    words1 = ['with tags','with tag','in box','in packet','bnwt','nwt','bnip',
              'nip','bnib','nib','mib','mip']
    words2 = ['without tags','with out tags','without tag','with out tag',
              'without box','with out box','without packet',
              'with out packet','bnwot','bnwob','nwot','mwob','mwop']
    if re.compile('|'.join(words1),re.IGNORECASE).search(txt):
        return 2
    elif re.compile('|'.join(words2),re.IGNORECASE).search(txt):
        return 1
    else: return 0
    
# def unknown_checker(txt):
#     return np.where(txt=='unknown',0,1)
print('Created required functions')    

merged['cat1'],merged['cat2'],merged['cat3'] = \
zip(*merged.category_name.apply(lambda x: Category_splitter(x)))
print('seperated category var into cat1,cat2,cat3')

Na_remover(merged)
print('removed NAs')

Cutting(merged)
print('cutting data------done')

start_time = time.time()
item_value = merged.item_description.apply(lambda x: desc_finder(x))
print('time taken by item value finder:{:.4f} minutes'.format((time.time()-start_time)*0.0166))

# brand_presence = merged.brand_name.apply(lambda x:unknown_checker(x))
# category_presence = merged.category_name.apply(lambda x:unknown_checker(x))
# description_presence = merged.item_description.apply(lambda x:unknown_checker(x))

Category_maker(merged)
print('converting into category dtype------done')


cv  = CountVectorizer(min_df=10,ngram_range=(1, 2),stop_words='english')
name_cv = cv.fit_transform(merged['name'])

cv = CountVectorizer()
cat1_cv = cv.fit_transform(merged.cat1)
cat2_cv = cv.fit_transform(merged.cat2)
cat3_cv = cv.fit_transform(merged.cat3)
print('countvectorizer on name and all cat`s--------done')


condition_dum = csr_matrix(pd.get_dummies(merged.item_condition_id,sparse=True).values)
print('dummies of condition----done')

lb = LabelBinarizer(sparse_output=True)
brand_binarized = lb.fit_transform(merged.brand_name)
print('NLP on name------done')

start_time = time.time()
tfid = TfidfVectorizer(max_features=100000,stop_words='english',ngram_range=(1,2))
description_tfid = tfid.fit_transform(merged.item_description)
print('NLP on description------done')
print('time taken:{:.4f} minutes'.format((time.time()-start_time)*0.0166))

tfid_mean = description_tfid.mean(1)
print('Making tfid_min variable------done')

shipping_sparse = csr_matrix(merged.shipping).T
item_value_sparse = csr_matrix(item_value).T
tfid_mean_sparse = csr_matrix(tfid_mean)
# presence_sparse = csr_matrix([brand_presence,category_presence,description_presence]).T

del merged
gc.collect()


X_sparse = hstack((name_cv,
            condition_dum,
            cat1_cv,cat2_cv,cat3_cv,
            brand_binarized,
            shipping_sparse,
            item_value_sparse,
            tfid_mean_sparse,
            description_tfid)).tocsr()
print('Creating sparse matrix---------done')

X_train = X_sparse[:Red_trn_len]
X_test = X_sparse[og_trn_len:]
del X_sparse
gc.collect()

start_time = time.time()
rmodel = Ridge(random_state=77,alpha=3,tol=0.01)
rmodel.fit(X_train,y_train)
predicted = rmodel.predict(X_test)
print('Model building---done')
print('time taken:{:.4f} minutes'.format((time.time()-start_time)*0.0166))

output = pd.DataFrame()
output['test_id'] = t_id
output['price'] = np.expm1(predicted)

output.to_csv('Ridge_hyperparam_and_ngram_maxfeature_changed.csv',index = False)