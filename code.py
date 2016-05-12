# coding= utf-8

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from time import time

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from matplotlib import pyplot as plt
# Load Data with pandas, and parse the first column into datetime
songs = pd.read_csv('..\\mars_tianchi_songs.csv', parse_dates=['publish_time'])
user_actions = pd.read_csv('..\\user_actions_add_artistid.csv', parse_dates=[
                           'Ds']).dropna(how='any')

# # Logistic Regression for comparison
# time1 = time()
# model2 = LogisticRegression(penalty='l2', C=.02,n_jobs = 4)
# model2.fit(training[features], training['crime'])
# predicted2 = np.array(model2.predict_proba(validation[features]))
# time2 = time()
# # print "Logistic Regression: ", time2 - time1
# log_loss(validation['crime'], predicted2)


# predicted = model4.predict_proba(test_data[features].fillna(value=0.0))

# # Write results
# result = pd.DataFrame(predicted, columns=le_crime.classes_)
# result.to_csv('testResult.csv', index=True, index_label='Id')

ids = ['3e395c6b799d3d8cb7cd501b4503b536', '0c80008b0a28d356026f4b1097041689', '25739ad1c56a511fcac86018ac4e49bb',
       '2e14d32266ee6b4678595f8f50c369ac', '5e2ef5473cbbdb335f6d51dc57845437', '6a493121e53d83f9e119b02942d7c8fe']
song_id = songs['song_id'].tolist()
artist_id = list(set(user_actions['artist_id'].tolist()))
# hehe = user_actions[user_actions.artist_id == '3e395c6b799d3d8cb7cd501b4503b536']
# hehe = user_actions[user_actions.artist_id == '0c80008b0a28d356026f4b1097041689']
# hehe = user_actions[user_actions.artist_id == '25739ad1c56a511fcac86018ac4e49bb']
# hehe = user_actions[user_actions.artist_id == '2e14d32266ee6b4678595f8f50c369ac']
# hehe = user_actions[user_actions.artist_id == '5e2ef5473cbbdb335f6d51dc57845437']
plt.ion()
for i in artist_id:
    hehe = user_actions[user_actions.artist_id == i]

    # hehe = hehe.set_index('Ds')
    data = []

    date = sorted(list(set(hehe['Ds'].tolist())))
    print date[0], date[-1]
    for d in date:
        data.append(hehe[hehe.Ds == d].shape[0])

    num = pd.Series(data)
    plt.figure()
    plt.suptitle("artitst:" + i)
    p1 = plt.subplot(311)
    p2 = plt.subplot(312)
    p3 = plt.subplot(313)

    window = 7
    p1.plot(num)
    p1.plot(pd.rolling_mean(num, window), color='red', label='mean')
    p1.plot(pd.rolling_std(num, window), color='green', label='std')

    diff1 = np.diff(num)
    p2.plot(diff1)
    p2.plot(pd.rolling_mean(diff1, window), color='red', label='mean')
    p2.plot(pd.rolling_std(diff1, window), color='green', label='std')

    diff2 = np.diff(num, n=2)
    p3.plot(diff2)
    p3.plot(pd.rolling_mean(diff2, window), color='red', label='mean')
    p3.plot(pd.rolling_std(diff2, window), color='green', label='std')
    plt.show()
    plt.savefig("..\\artist_" + i + ".jpg")
# 为user_actions 增加一列artist_id
# artist_id = []

# song_artist = {}

# for i in range(songs.shape[0]):
#     song_artist[songs.loc[i]['song_id']] = songs.loc[i]['artist_id']

# for song in user_actions['song_id']:
#     artist_id.append(song_artist[song])
