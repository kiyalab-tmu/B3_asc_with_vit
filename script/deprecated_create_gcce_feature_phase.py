import numpy as np
import librosa
from tqdm import tqdm
import os
import pickle
import cv2
import colorsys
import cmath
import math


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
    feature_name = 'phase_batch' + batch_size
    """
    """

    # このファイル自体の内容を取得
    filename = os.path.splitext(os.path.basename(__file__))[0]
    with open(filename + '.py', mode='r') as f:
        code_contents = f.read()

    # Dataframeの読み込み
    with open('../output/' + dataset_name + '/dataframe/train/dataframe.pickle', 'rb') as f:
        train_df = pickle.load(f)
    with open('../output/' + dataset_name + '/dataframe/valid/dataframe.pickle', 'rb') as f:
        valid_df = pickle.load(f)
    with open('../output/' + dataset_name + '/dataframe/test/dataframe.pickle', 'rb') as f:
        test_df = pickle.load(f)



    # 保存先のディレクトリがなかったら作る
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/train/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/train/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/test/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/test/')


    def calc_ampli_and_phase(wave):
        stft_output = librosa.stft(wave, n_fft=window_length, hop_length=hop_length, window='hann', center=False)
        for i in range(stft_output.shape[0]):
            for j in range(stft_output.shape[1]):
                re = stft_output[i][j].real
                im = stft_output[i][j].imag
                if re == 0:
                    stft_output[i][j] = 0
                else:
                    stft_output[i][j] = math.atan(im/re)
        stft_output = stft_output.real

        amplitude = np.abs(librosa.stft(wave, n_fft=window_length, hop_length=hop_length, window='hann', center=False))
        amplitude = np.log(amplitude + 0.0001)

        output = []
        output.append(amplitude)
        output.append(amplitude)
        output.append(stft_output)
        output = np.array(output)
        output = output.transpose(1,2,0)

        return output


    def create_one_sample(path):
        wave, _ = librosa.load(path, mono=True, sr=16000)
        phase = calc_ampli_and_phase(wave)


        phase = phase[:pre_image_size, :pre_image_size] # 画像サイズを496,496に設定
        phase = cv2.resize(phase, dsize=(image_size, image_size)) #サイズを小さくする
    
        return phase


    
    def do_create_feature(df, data_type):
        feature_data = []
        label_data = []
        for i in tqdm(range(df.shape[0])):
            returned_data = create_one_sample(path=df['path'].iloc[i])
            feature_data.append(returned_data)
            label_data.append(df['label'].iloc[i])
                
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
    do_create_feature(df=train_df, data_type='train')
    print('######Valid######')
    do_create_feature(df=valid_df, data_type='valid')
    print('######Test######')
    do_create_feature(df=test_df, data_type='test')
    

