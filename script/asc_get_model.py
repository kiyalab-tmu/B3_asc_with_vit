from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from vit_keras import vit, utils

def get_my_model(model_info, class_num, **kwargs):
    if model_info == 'resnet50':
        base_resnet50 = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=kwargs['input_shape'], pooling='avg')
        x = base_resnet50.output
        predictions = Dense(class_num, activation='softmax')(x)
        model = keras.Model(inputs=base_resnet50.input, outputs=predictions)
        return model



    elif model_info == 'dcase_ibm':
        model = Sequential()
        model.add(Conv2D(64, (7, 1), padding='same', input_shape=kwargs['input_shape']))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(4, 1)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (10, 1), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(256, (1, 7), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(1, 256)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128))

        model.add(Dense(class_num, activation='softmax'))
        return model


        
    elif model_info == 'vit':
        model = vit.vit_b16(
            image_size = kwargs['image_size'],
            activation = 'sigmoid',
            pretrained = True,
            include_top = True,
            pretrained_top = False,
            classes=class_num)
        return model
    


if __name__ == '__main__':
    model = get_my_model(model_info='vit', class_num=9, image_size=496)
    print(model.summary())
