import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import io
from matplotlib import pyplot as plt






if __name__ == '__main__':
    # ==========
    # 手動設定事項
    # ==========
    is_save = True




    # その他の設定
    dataset_name = 'sinsdcasenode2'
    top_path = '/home/shiroma7/my_sins/audio/10s_10s/Node2/'
    save_dir_path = '../output/' + dataset_name + '/dataframe/'
    
    
    # ファイル名やラベルの情報が入った.mファイルをdataframeへ変換
    matdata = io.loadmat('../source/living_segment_10s_10slabels.mat', squeeze_me=True)["label_info"]
    df = pd.DataFrame(data=matdata, columns=['path', 'label', 'segment'])
    df = df[df['label'] != 'dont use']
    df.loc[df['label'] == 'watching tv', 'label'] = 'watching_tv'
    df.loc[df['label'] == 'calling', 'label'] = 'social_activity'
    df.loc[df['label'] == 'visit', 'label'] = 'social_activity'

    # ファイル名に正しいファイルパスを付加する
    df['path'] = top_path + df['path']
    print(df['label'].unique())

    #このプログラム自体の内容を取得
    filename = os.path.splitext(os.path.basename(__file__))[0]
    with open(filename + '.py', mode='r') as f:
        code_contents = f.read()

    
    # 各ラベルごとのdataframeを作る。
    # また、ラベル名をDCASE2018Task5のデータと同じになるように設定
    # ['absence' 'other' 'working' 'cooking' 'eating' 'social_activity' 'dishwashing' 'vacuumcleaner' 'watching tv']
    df_absence = df[df['label'] == 'absence']
    df_other = df[df['label'] == 'other']
    df_working = df[df['label'] == 'working']
    df_cooking = df[df['label'] == 'cooking']
    df_eating = df[df['label'] == 'eating']
    df_social_activity = df[df['label'] == 'social_activity']
    df_dishwashing = df[df['label'] == 'dishwashing']
    df_vacuumcleaner = df[df['label'] == 'vacuumcleaner']
    df_watching_tv = df[df['label'] == 'watching_tv']
    
    # (各ラベルごとに)データ数をDCASE2018Task5のデータと同じになるように設定
    df_absence = df_absence[0:4715]
    df_other = df_other[0:515]
    df_working = df_working[0:4661]
    df_cooking = df_cooking[0:1281]
    df_eating = df_eating[0:577]
    df_social_activity = df_social_activity[0:1236]
    df_dishwashing = df_dishwashing[0:356]
    df_vacuumcleaner = df_vacuumcleaner[0:243]
    df_watching_tv = df_watching_tv[0:4661]




    
    # (各ラベルごとに)データフレームを train&valid と testに8:2で分割
    df_absence_train, df_absence_test = train_test_split(df_absence, test_size=0.2, shuffle = False)
    df_other_train, df_other_test = train_test_split(df_other, test_size=0.2, shuffle = False)
    df_working_train, df_working_test = train_test_split(df_working, test_size=0.2, shuffle = False)
    df_cooking_train, df_cooking_test = train_test_split(df_cooking, test_size=0.2, shuffle = False)
    df_eating_train, df_eating_test = train_test_split(df_eating, test_size=0.2, shuffle = False)
    df_social_activity_train, df_social_activity_test = train_test_split(df_social_activity, test_size=0.2, shuffle = False)
    df_dishwashing_train, df_dishwashing_test = train_test_split(df_dishwashing, test_size=0.2, shuffle = False)
    df_vacuumcleaner_train, df_vacuumcleaner_test = train_test_split(df_vacuumcleaner, test_size=0.2, shuffle = False)
    df_watching_tv_train, df_watching_tv_test = train_test_split(df_watching_tv, test_size=0.2, shuffle = False)
    trainvalid_df = pd.concat([df_absence_train, df_other_train, df_working_train, df_cooking_train, df_eating_train, df_social_activity_train, df_dishwashing_train, df_vacuumcleaner_train, df_watching_tv_train], axis=0)
    test_df = pd.concat([df_absence_test, df_other_test, df_working_test, df_cooking_test, df_eating_test, df_social_activity_test, df_dishwashing_test, df_vacuumcleaner_test, df_watching_tv_test], axis=0)


    # データフレームを train と valid に8:2で分割
    train_df, valid_df = train_test_split(trainvalid_df,  test_size=0.2, random_state=0, stratify=trainvalid_df['label'])

    # データフレーム内をシャッフル
    train_df = train_df.sample(frac=1)
    valid_df = valid_df.sample(frac=1)
    test_df = test_df.sample(frac=1)
    # データフレームのindexを整える
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)


    # 最終的に完成したdataframeを表示
    print(train_df)
    print(train_df['label'].value_counts())
    print(valid_df)
    print(valid_df['label'].value_counts())
    print(test_df)
    print(test_df['label'].value_counts())


    if is_save:
        # ディレクトリなかったら作る
        if not os.path.isdir('../' + 'output/'):
            os.makedirs('../' + 'output/')
        if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/'):
            os.makedirs('../' + 'output/' + dataset_name + '/dataframe/')
        if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/train/'):
            os.makedirs('../' + 'output/' + dataset_name + '/dataframe/train/')
        if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/valid/'):
            os.makedirs('../' + 'output/' + dataset_name + '/dataframe/valid/')
        if not os.path.isdir('../' + 'output/' + dataset_name + '/dataframe/test/'):
            os.makedirs('../' + 'output/' + dataset_name + '/dataframe/test/')

        #保存
        with open(save_dir_path + 'train/dataframe.pickle', 'wb') as f:
            pickle.dump(train_df, f, protocol=4)
        with open(save_dir_path + 'valid/dataframe.pickle', 'wb') as f:
            pickle.dump(valid_df, f, protocol=4)
        with open(save_dir_path + 'test/dataframe.pickle', 'wb') as f:
            pickle.dump(test_df, f, protocol=4)
        with open(save_dir_path + 'test/memo.txt', 'w') as f:
            f.write(code_contents)




    
