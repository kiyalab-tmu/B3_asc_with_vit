import numpy as np
import librosa
from tqdm import tqdm
import os
import pickle
import cv2


if __name__ == '__main__':
    # ==============
    # 必須手動設定項目
    # ==============
    dataset_name = 'gccensynth'


    # ==========
    # 手動設定項目
    # ==========
    window_length = 1024
    hop_length = 125
    batch_size = 32
    pre_image_size = 496 # 元々は(513, 504)。端っこはすてる。
    image_size = 224
    feature_name = 'logmelspectrogram3ch_batch' + str(batch_size)
    """
    """

    # このファイル自体の内容を取得
    filename = os.path.splitext(os.path.basename(__file__))[0]
    with open(filename + '.py', mode='r') as f:
        code_contents = f.read()

    # ファイル名と正解ラベルのリストの読み込み
    with open('../../nsynth_data/train/filename_and_label.pickle', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('../../nsynth_data/valid/filename_and_label.pickle', 'rb') as f:
        valid_dataset = pickle.load(f)
    with open('../../nsynth_data/test/filename_and_label.pickle', 'rb') as f:
        test_dataset = pickle.load(f)



    # 保存先のディレクトリがなかったら作る
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/train/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/train/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/test/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/test/')


    def calc_logspectrogram(wave):
        #スペクトログラムの生成
        spectrogram = np.abs(librosa.stft(wave, n_fft=window_length, hop_length=hop_length, center=False))
        #ログスペクトログラムにする
        logspectrogram = np.log(spectrogram + 0.0001)
        logspectrogram = logspectrogram[:pre_image_size, :pre_image_size] # 画像サイズを496,496に設定
        logspectrogram = cv2.resize(logspectrogram, dsize=(image_size, image_size)) #サイズを小さくする
        return logspectrogram


    def create_one_sample(path):
        one_sample = list()
        wave, _ = librosa.load(path, mono=True, sr=16000)
        logspectrogram = calc_logspectrogram(wave)
        

        one_sample.append(logspectrogram)
        one_sample.append(logspectrogram) # 3chに増幅
        one_sample.append(logspectrogram) # 3chに増幅

        one_sample = np.array(one_sample)
        one_sample = one_sample.transpose(1,2,0)
        return one_sample


    
    def do_create_feature(dataset, data_type):
        feature_data = []
        label_data = []
        for i in tqdm(range(len(dataset))):
            returned_data = create_one_sample(path='../../nsynth_data/' + data_type + '/' + dataset[i][0])
            feature_data.append(returned_data)
            label_data.append(dataset[i][1])
                
            if len(feature_data) == batch_size:
                feature_data = np.array(feature_data)
                with open('../' + 'output/' + dataset_name + '/' + feature_name + '/' + data_type + '/feature_' + str(len(label_data)//batch_size) + '.pickle', 'wb') as f:
                    pickle.dump(feature_data, f)
                feature_data = list()
        # labelの保存
        label_data = np.array(label_data)
        with open('../' + 'output/' + dataset_name + '/' + feature_name + '/' + data_type + '/' + '/label.pickle', 'wb') as f:
            pickle.dump(label_data, f)
        #コードの保存
        with open('../' + 'output/' + dataset_name + '/' + feature_name + '/' + data_type + '/' + '/memo.txt', 'w') as f:
            f.write(code_contents)

    
    
    #実行!
    print('######Train######')
    do_create_feature(dataset=train_dataset, data_type='train')
    print('######Valid######')
    do_create_feature(dataset=valid_dataset, data_type='valid')
    print('######Test######')
    do_create_feature(dataset=test_dataset, data_type='test')
    

