import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

courses_df1 = pd.read_csv('ml-1m/courses.dat', sep='::', header=None, engine='python')
print(courses_df1.head())
courses_df1.to_csv('courses1.csv')
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python')
ratings_df.to_csv('ratings.csv')

courses_df1.columns = ['courseId', 'name', 'Genres']
ratings_df.columns = ['userId', 'courseId', 'rating', 'Timestamp']

print(courses_df1.head())
print(ratings_df.head())

print('The Number of coursess in Dataset', len(courses_df1))

courses_df1['List Index'] = courses_df1.index
print(courses_df1.head())

merged_df = courses_df1.merge(ratings_df, on='courseId')

merged_df = merged_df.drop('Timestamp', axis=1).drop('name', axis=1).drop('Genres', axis=1)

print(merged_df.head())

user_Group = merged_df.groupby('userId')
print(user_Group.head())

s = 1000 #amount of users

trainX = []

for userId, curUser in user_Group:

    temp = [0]*len(courses_df1)

    for num, c1 in curUser.iterrows():

        temp[c1['List Index']] = c1['rating']/5.0

    trainX.append(temp)

    if s == 0:
        break
    s -= 1
print(trainX)

testX = []
s1 = 2000
for userId, curUser in user_Group:

    temp1 = [0]*len(courses_df1)

    for num, c1 in curUser.iterrows():

        temp1[c1['List Index']] = c1['rating']/5.0

    trainX.append(temp)

    if s1== 1000:
        break
    s -= 1
print(testX)

hiddenUnits = 50
visibleUnits = len(courses_df1)
vb = tf.placeholder(tf.float32, [visibleUnits]) 
hb = tf.placeholder(tf.float32, [hiddenUnits])  
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  

v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling

_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

#RBM train parameters

alpha = 1.0

w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

err = tf.abs(v0 - v1)
err_abs = tf.reduce_mean(err)
loss_err = tf.reduce_mean(err*err)
err_rmse = tf.reduce_mean(tf.sqrt(err*err))
#err_rmse = tf.rmse
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

cur_vb = np.zeros([visibleUnits], np.float32)

cur_hb = np.zeros([hiddenUnits], np.float32)

prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

prv_vb = np.zeros([visibleUnits], np.float32)

prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 15
batchsize = 100
abs_error = []
rmse_errors = []
loss = [] 
acc = []
for i in range(epochs):
    for start, end in zip(range(0, len(trainX), batchsize), range(batchsize, len(trainX), batchsize)):
        batch = trainX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    abs_error.append(sess.run(err_abs, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    rmse_errors.append(sess.run(err_rmse, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    loss.append(sess.run(loss_err, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    
    acc.append(1-loss[i])
    print("loss: "+str(abs_error[-1])+" rmse : "+str(rmse_errors[-1]))
    
    print("accuracy: %.2f",acc[-1])
plt.plot(abs_error)
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.show()
   
print (acc)
plt.plot(acc)
plt.ylabel('Training Accuracy')
plt.xlabel('Epoch')
plt.show()

epochs = 15
batchsize = 100
abs_error = []
rmse_errors = [] 
acc = []
for i in range(epochs):
    for start, end in zip(range(0, len(testX), batchsize), range(batchsize, len(testX), batchsize)):
        batch = testX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    rmse_errors.append(sess.run(err_rmse, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    abs_error.append(sess.run(err_abs, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    loss.append(sess.run(loss_err, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))

    acc.append(1-loss[i])
    print("loss: "+str(abs_error[-1])+" rmse : "+str(rmse_errors[-1]))
    
    print("accuracy: ",acc[-1])
plt.plot(abs_error)
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.show()
   
print (acc)
plt.plot(acc)
plt.ylabel('validation Accuracy')
plt.xlabel('Epoch')
plt.show()

inputUser = [trainX[42]]
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

score_courses_df1_50 = courses_df1
score_courses_df1_50["Recommendation Score"] = rec[0]
print(score_courses_df1_50.sort_values(["Recommendation Score"], ascending=False).head(20))

""" Recommend User what courses he has not done yet """

print(merged_df.iloc[50])

courses_df1_50 = merged_df[merged_df['userId'] == 150]
print(courses_df1_50.head())


recommendation = score_courses_df1_50.merge(courses_df1_50, on='courseId', how='outer')

recommendation = recommendation.drop('List Index_y', axis=1).drop('userId', axis=1)
recommendation = recommendation.drop('rating',axis =1)


print(recommendation.sort_values(['Recommendation Score'], ascending=False).head(10))

