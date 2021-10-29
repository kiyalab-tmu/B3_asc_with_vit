import numpy as np
import librosa
from tqdm import tqdm
import os
import pickle
import cv2



if __name__ == '__main__':
    # =============
    # 必須手動設定項目
    # =============
    dataset_name = 'sinsdcasenode2'





    # ==========
    # 手動設定項目
    # ==========
    window_length = 1024
    hop_length = 312
    batch_size = 32
    pre_image_size = 496 # 元々は(510, 510)くらい。端っこはすてる。
    image_size = 224
    feature_name = 'logmelspectrogram3ch'
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



    # 保存先ディレクトリがなかったら作る
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/train/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/train/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/valid/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/' + feature_name + '/test/'):
        os.makedirs('../' + 'output/' + dataset_name + '/' + feature_name + '/test/')

    def calc_log_mel(wave):
        # 音声波形から直流成分除去
        wave = wave - np.mean(wave)
        #スペクトログラムの生成
        spectrogram = np.abs(librosa.stft(wave, n_fft=window_length, hop_length=hop_length, center=False))
        #ログスペクトログラムにする
        logspectrogram = np.log(spectrogram + 0.0001)
        logspectrogram = logspectrogram[:pre_image_size, :pre_image_size] # 画像サイズを496,496に設定
        logspectrogram = cv2.resize(logspectrogram, dsize=(image_size, image_size)) #サイズを小さくする
        return logspectrogram


    def create_one_sample(path):
        # Node2のpathから、Node4,6のPathを生成
        path1 = path.replace('Node2', 'Node2')
        path2 = path.replace('Node2', 'Node4')
        path3 = path.replace('Node2', 'Node6')

        #Node1
        one_sample = list()
        wave, _ = librosa.load(path1, mono=True, sr=16000)
        logmelspectrogram = calc_log_mel(wave)
        one_sample.append(logmelspectrogram)
    
        #Node2
        wave, _ = librosa.load(path2, mono=True, sr=16000)
        logmelspectrogram = calc_log_mel(wave)
        one_sample.append(logmelspectrogram)


        #Node3
        wave, _ = librosa.load(path3, mono=True, sr=16000)
        logmelspectrogram = calc_log_mel(wave)
        one_sample.append(logmelspectrogram)




        one_sample = np.array(one_sample)
        one_sample = one_sample.transpose(1,2,0)

        return one_sample


        
    
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

    
    
    #実行！
    print('######Train######')
    do_create_feature(df=train_df, data_type='train')
    print('######Valid######')
    do_create_feature(df=valid_df, data_type='valid')
    print('######Test######')
    do_create_feature(df=test_df, data_type='test')
    

