#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import json
import os


if __name__ == '__main__':
    # ==========
    # 手動設定項目
    # ==========
    dataset_name = 'gccensynth'
    top_path = '/lhome/audio/NSynth/'
    json_path_train = '/home/audio/NSynth/nsynth-train/examples.json'
    json_path_valid = '/home/audio/NSynth/nsynth-valid/examples.json'
    json_path_test  = '/home/audio/NSynth/nsynth-test/examples.json'



    # このファイル自体の内容を取得
    filename = os.path.splitext(os.path.basename(__file__))[0]
    with open(filename + '.py', mode='r') as f:
        code_contents = f.read()

    # 各種データの読み込み
    save_dir_path = '../output/' + dataset_name + '/dataframe/'
    train_dict = json.load(open(json_path_train , 'r'))
    valid_dict = json.load(open(json_path_valid , 'r'))
    test_dict = json.load(open(json_path_test , 'r'))
    train_df = pd.DataFrame.from_dict(train_dict, orient='index')
    valid_df = pd.DataFrame.from_dict(valid_dict, orient='index')
    test_df = pd.DataFrame.from_dict(test_dict, orient='index')



    def operate_dataframe(df):
        # いらないデータを消去
        del df['pitch']
        del df['note']
        del df['velocity']
        del df['instrument_str']
        del df['instrument']
        del df['sample_rate']
        del df['note_str']
        del df['instrument_source']
        del df['instrument_family']
        del df['qualities']
        del df['qualities_str']
        # Dataframeの列のタイトルなどを整える
        df = df.rename_axis('path').reset_index()
        df = df.iloc[:, [0,2,1]]
        df.rename(columns={'instrument_family_str':'label', 'instrument_source_str':'source'}, inplace=True)
        df['path'] = df['path'] + '.wav'
        # いらないデータを削除
        df = df[df['source'] != 'synthetic']
        df = df[df['label'] != 'vocal']
        df = df[df['label'] != 'flute']
        return df


    # 作業する
    train_df = operate_dataframe(train_df)
    valid_df = operate_dataframe(valid_df)
    test_df = operate_dataframe(test_df)
    
    #シャッフルする
    train_df = train_df.sample(frac=1)
    valid_df = valid_df.sample(frac=1)
    test_df = test_df.sample(frac=1)


    #データ数を制限
    def limit_data(df, limit_num):
        keyboard_counter = 0
        guitar_counter   = 0
        organ_counter    = 0
        string_counter   = 0
        brass_counter    = 0
        reed_counter     = 0
        mallet_counter   = 0
        bass_counter     = 0
        new_df = df[:0]
        for i in range(len(df)):
            if   df['label'].iloc[i] == 'keyboard' and keyboard_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                keyboard_counter = keyboard_counter + 1
            elif df['label'].iloc[i] == 'guitar' and guitar_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                guitar_counter = guitar_counter + 1
            elif df['label'].iloc[i] == 'organ' and organ_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                organ_counter = organ_counter + 1
            elif df['label'].iloc[i] == 'string' and string_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                string_counter = string_counter + 1
            elif df['label'].iloc[i] == 'brass' and brass_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                brass_counter = brass_counter + 1
            elif df['label'].iloc[i] == 'reed' and reed_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                reed_counter = reed_counter + 1
            elif df['label'].iloc[i] == 'mallet' and mallet_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                mallet_counter = mallet_counter + 1
            elif df['label'].iloc[i] == 'bass' and bass_counter < limit_num:
                new_df = pd.concat([new_df, df[i:i+1]])
                bass_counter = bass_counter + 1

        return new_df
    
    # データ数を制限する
    train_df = limit_data(train_df, 5000).reset_index(drop=True)
    valid_df = limit_data(valid_df, 470).reset_index(drop=True)
    test_df = limit_data(test_df, 500).reset_index(drop=True)

    # ファイル名に正しいファイルパスを付加する
    train_df['path'] = top_path + 'nsynth-train/audio/' + train_df['path']
    valid_df['path'] = top_path + 'nsynth-valid/audio/' + valid_df['path']
    test_df['path'] = top_path + 'nsynth-test/audio/' + test_df['path']

    # 確認用
    print(train_df['label'].value_counts())
    print(valid_df['label'].value_counts())
    print(test_df['label'].value_counts())
    # 確認用
    print(train_df)
    print(valid_df)
    print(test_df)


    # ディレクトリなかったら作る
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/train/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/train/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/valid/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/valid/')
    if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/test/'):
        os.makedirs('../' + 'output/' + dataset_name + '/dataframe/test/')

    # 保存
    with open(save_dir_path + 'train/dataframe.pickle', 'wb') as f:
        pickle.dump(train_df, f, protocol=4)
    with open(save_dir_path + 'valid/dataframe.pickle', 'wb') as f:
        pickle.dump(valid_df, f, protocol=4)
    with open(save_dir_path + 'test/dataframe.pickle', 'wb') as f:
        pickle.dump(test_df, f, protocol=4)
    
    # コードの保存
    with open(save_dir_path + 'test/memo.txt', 'w') as f:
        f.write(code_contents)

