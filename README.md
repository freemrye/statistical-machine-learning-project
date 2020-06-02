# statistical-machine-learning-project  Yixin Ye

### import package
import numpy as np / 
import pandas as pd
import lightgbm as lgb

### import data
data_path = 'E:/FinTech/Statistical machine learning/CourseProject/music recommendation/'
train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category','source_system_tab' : 'category',
                                                    'source_screen_name' : 'category','source_type' : 'category',
                                                  'target' : np.uint8,'song_id' : 'category'})
test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category', 'source_system_tab' : 'category',
                                                'source_screen_name' : 'category', 'source_type' : 'category',
                                                'song_id' : 'category'})
songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category','language' : 'category',
                                                  'artist_name' : 'category','composer' : 'category',
                                                  'lyricist' : 'category','song_id' : 'category'})
members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category','bd' : np.uint8,
                                                      'gender' : 'category','registered_via' : 'category'})
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

print(train.shape)
print(train['msno'].value_counts())


### data mining
#merge data
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

train.head()


#deal with 'time' in members
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)

songs_extra

#deal with 'isrc' in songs_extra
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

#merge
train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')

train

#datatype
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
X = train.drop(['target'], axis=1)
y = train['target'].values

### Train the model
#split the data into training set and test set
X_train = X[:5000000]
y_train = y[:5000000]
X_val = X[5000000:]
y_val = y[5000000:]

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

d_train = lgb.Dataset(X_train, y_train)
d_val = lgb.Dataset(X_val, y_val)

#train the model
params = {}
params['learning_rate'] = 0.5
params['application'] = 'binary'
params['max_depth'] = 3
params['num_leaves'] = 50
params['metric'] = 'auc'

model = lgb.train(params, train_set=d_train, num_boost_round=2000, valid_sets=[d_val], \
verbose_eval=100)

### making prediction
p_test = model.predict(X_test)
submission = pd.DataFrame()
submission['id'] = ids
submission['target'] = p_test
submission.to_csv('submission.csv')
