#!/usr/bin/env python
# coding: utf-8

# =======
# 初期設定
# =======
import sys
import datetime
import hashlib
from os import path
print('実験ディレクトリを入力:')
result_dir_name = input()
# print('GPU idを入力:')
# gpu_id = int(input())
ex_name = 'asc_' + hashlib.md5(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode()).hexdigest()
print('今回の実験ID')
filename = path.splitext(path.basename(__file__))[0]
with open(filename + '.py', mode='r') as f:
    code_contents = f.read()




# ==========
# 手動設定項目
# ==========
batch_size = 32 # ここを変える時はcreate_***_feature_***.pyの変数も変えて、特徴量の再抽出が必要
learning_rate = 0.0001
epoch_num = 10
dataset_name = 'gccensynth' # 'sinsdcasenode2' か 'gccensynth' か 'mnist'
feature_name = 'logmelspectrogram3ch_batch32'
image_size = 224 # 固定
model_name = 'resnet50' # 'vit' か 'resnet50'
class_num = 4 # dataset_nameが sinsdcasenode2なら9, gccensynthなら4, mnistなら10
fine_tuning = None # None か 'imagenet' (モデルがResNet50のときのみ有効。ViTは常にimagenet)
def optional_function(input):
    return input






# ==========
# 自動設定項目
# ==========
print('プログラム読み込み完了')
print(ex_name)
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from asc_my_audio_data_generator import my_audio_datagenerator
from asc_get_model import get_my_model
train_label_path = '../output/' + dataset_name + '/' + feature_name + '/train/label.pickle'
train_data_str   = '../output/' + dataset_name + '/' + feature_name + '/train/feature_'
valid_label_path = '../output/' + dataset_name + '/' + feature_name + '/valid/label.pickle'
valid_data_str   = '../output/' + dataset_name + '/' + feature_name + '/valid/feature_'
test_label_path  = '../output/' + dataset_name + '/' + feature_name + '/test/label.pickle'
test_data_str    = '../output/' + dataset_name + '/' + feature_name + '/test/feature_'


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
# tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)



# =========================
# create generator instance
# =========================
train_datagen = my_audio_datagenerator()
valid_datagen = my_audio_datagenerator()
test_datagen  = my_audio_datagenerator()






# ===============
# モデルのコンパイル
# ===============
# model_name='vit'なら get_my_model(model_info=model_name, class_num=class_num, image_size=image_size)
# model_name='resnet50'なら get_my_model(model_info=model_name, class_num=class_num, input_shape=(224,224,3))
model = None
if model_name=='vit':
    model = get_my_model(model_info=model_name, class_num=class_num, image_size=image_size)
elif model_name=='resnet50':
    model = get_my_model(model_info=model_name, class_num=class_num, input_shape=(224,224,3), fine_tuning = fine_tuning)



adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['acc'])
print(model.summary())







# ====================
# 必要なディレクトリの生成
# ====================
if not os.path.isdir('../' + 'result/'):
    os.makedirs('../' + 'result/')
if not os.path.isdir('../' + 'result/' + result_dir_name + '/'):
    os.makedirs('../' + 'result/' + result_dir_name + '/')
if not os.path.isdir('../' + 'result/' + result_dir_name + '/models/'):
    os.makedirs('../' + 'result/' + result_dir_name + '/models/')
if not os.path.isdir('../' + 'result/' + result_dir_name + '/result/'):
    os.makedirs('../' + 'result/' + result_dir_name + '/result/')






# ====================
# ラベルの読み込み
# ====================
with open(train_label_path, 'rb') as f:
    train_label = pickle.load(f)
with open(valid_label_path, 'rb') as f:
    valid_label = pickle.load(f)
with open(test_label_path, 'rb') as f:
    test_label = pickle.load(f)




# ====
# 学習
# ====
checkpoint = tf.keras.callbacks.ModelCheckpoint('../' + 'result/' + result_dir_name + '/models/' + ex_name +'_model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
history = model.fit(
    x=train_datagen.flow_from_pickle(batch_size=batch_size, label_path=train_label_path, data_str=train_data_str, func=optional_function, isRandom=True),
    steps_per_epoch = len(train_label) // batch_size,
    batch_size=batch_size,
    epochs=epoch_num,
    verbose=2,
    validation_data=valid_datagen.flow_from_pickle(batch_size=batch_size, label_path=valid_label_path, data_str=valid_data_str, func=optional_function, isRandom=True),
    validation_steps= len(valid_label) // batch_size,
    shuffle=False,
    callbacks=[checkpoint])



# ===================
# ラストエポック後の結果
# ===================
print('loss:', history.history['loss'][-1])
print('val_loss:', history.history['val_loss'][-1])
print('acc:', history.history['acc'][-1])
print('val_acc:', history.history['val_acc'][-1])









# =================
# 学習結果のグラフ表示
# =================
# Loss
plt.figure(figsize=(8,8))
plt.title('Loss Value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
# plt.show()
plt.savefig('../' + 'result/' + result_dir_name + '/result/' + ex_name + "_loss_result" + ".png")
plt.clf()

# Accuracy
plt.figure(figsize=(8,8))
plt.title('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.xlim(0, epoch_num) 
plt.ylim(0, 1.0)
# plt.show()
plt.savefig('../' + 'result/' + result_dir_name + '/result/' + ex_name + "_acc_result" + ".png")
plt.clf()









# ===========================
# 全学習の中の最良モデルでのテスト
# ===========================
# 一番良かったモデルを読み込む
model = None
model = tf.keras.models.load_model('../' + 'result/' + result_dir_name + '/models/' + ex_name +'_model.h5')
# 評価
train_loss, train_acc = model.evaluate(train_datagen.flow_from_pickle(batch_size=batch_size, label_path=train_label_path, data_str=train_data_str, func=optional_function, isRandom=True), verbose=0, steps=len(train_label) // batch_size)
valid_loss, valid_acc = model.evaluate(valid_datagen.flow_from_pickle(batch_size=batch_size, label_path=valid_label_path, data_str=valid_data_str, func=optional_function, isRandom=True), verbose=0, steps=len(valid_label) // batch_size)
test_loss, test_acc = model.evaluate(test_datagen.flow_from_pickle(batch_size=batch_size, label_path=test_label_path, data_str=test_data_str, func=optional_function, isRandom=True), verbose=0, steps=len(test_label) // batch_size)
print('\n\n最良モデルで評価\n')
print('train_loss:', train_loss)
print('valid_loss:', valid_loss)
print('test_loss:', test_loss)
print('train_acc:', train_acc)
print('valid_acc:', valid_acc)
print('test_acc:', test_acc)
print('ex_name:', ex_name)




# ==============
# F1-scoreを抽出
# ==============
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.utils import to_categorical
predictions = model.predict(test_datagen.flow_from_pickle(batch_size=batch_size, label_path=test_label_path, data_str=test_data_str, func=optional_function, isRandom=False), verbose=0, steps=len(test_label) // batch_size) #softmax出力のまま。各クラスに属する確率
y_pred = np.argmax(predictions, axis=1)
with open(test_label_path, 'rb') as f:
    test_labels = pickle.load(f)
classes = np.unique(test_labels)
classes = {v: i for i, v in enumerate(sorted(classes))}
label_tamesi = []
for i in range(len(test_labels)):
    label_tamesi.append(classes[test_labels[i]])
f1 = f1_score(label_tamesi[:(len(test_label) // batch_size) * batch_size], y_pred , average="macro")
print('f1_score:', f1)







# ======================
# confusion matrixを生成
# ======================
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(test_label[:(len(test_label) // batch_size) * batch_size])
label_encoded = label_encoded[:, np.newaxis]
one_hot_encoder = OneHotEncoder(sparse=False)
test_one_hot_vector = one_hot_encoder.fit_transform(label_encoded)
y_test = one_hot_encoder.inverse_transform(test_one_hot_vector) #one hotをもとに戻す

cm_yoko_percent = confusion_matrix(label_tamesi[:(len(test_label) // batch_size) * batch_size], y_pred)
cm_yoko_percent = np.array(cm_yoko_percent, dtype='f4')
print(cm_yoko_percent.shape)
for i in range(cm_yoko_percent.shape[0]):
    row_sum = np.sum(cm_yoko_percent[i])
    for j in range(cm_yoko_percent.shape[1]):
        if row_sum == 0:
            cm_yoko_percent[i][j] = 0
        else:
            cm_yoko_percent[i][j] = cm_yoko_percent[i][j] / row_sum * 100


# 保存
plt.figure(figsize=(12,12))
sns.heatmap(cm_yoko_percent, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# plt.show()
plt.savefig('../' + 'result/' + result_dir_name + '/result/' + ex_name + "_model_recall_result" + ".png")
plt.clf()






# =========
# メモを残す
# =========
with open('../' + 'result/' + result_dir_name + '/result/' + ex_name + "_memo" + ".txt", mode='w') as f:
    f.write('\ntrain_loss:' + str(train_loss))
    f.write('\nvalid_loss:' + str(valid_loss))
    f.write('\ntest_loss:' + str(test_loss))
    f.write('\ntrain_acc:' + str(train_acc))
    f.write('\nvalid_acc:' + str(valid_acc))
    f.write('\ntest_acc:' + str(test_acc))
    f.write('\nf1-score:' + str(f1))
    f.write('\n\n\n\n\n実際のコード\n\n\n\n')


with open('../' + 'result/' + result_dir_name + '/result/' + ex_name + "_memo" + ".txt", mode='a') as f:
    print(code_contents, file=f)
    
    
    
    
"""
追加
""" 
print(y_pred) #テストデータを上から順番に(nsynth_data/test/filename_and_label.pickleの)予測した結果。model.predictはf-scoreの計算のところですでに実施済み。
"""
追加
""" 
