import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
import random




class my_audio_datagenerator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = []

    def flow_from_pickle(self, batch_size, label_path, data_str, func, isRandom):
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
        classes = np.unique(labels)
        classes = {v: i for i, v in enumerate(sorted(classes))}
        while True:
            index_list = list(range(len(labels) // batch_size))
            if isRandom:
                random.shuffle(index_list)
            for i in range(len(labels) // batch_size):
                batch_data = None
                with open(data_str + str(index_list[i]+1) + '.pickle', 'rb') as f :
                    batch_data = pickle.load(f)
                

                if func != None:
                    batch_data = func(batch_data)

                



                tmp_labels = labels[index_list[i]*batch_size:index_list[i]*batch_size + batch_size]
                label_one_hot = []
                for j in range(len(tmp_labels)):
                    label_one_hot.append(to_categorical(classes[tmp_labels[j]], len(classes)))
                label_one_hot = np.array(label_one_hot)


                yield batch_data, label_one_hot

                
                
