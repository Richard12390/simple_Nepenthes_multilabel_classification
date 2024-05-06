import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from hyperopt import Trials, tpe
from hyperas import optim
from hyperas.distributions import choice, quniform
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

def prepare_data():
    IMG_DIR = os.path.join("input","train")
    
    for fold_list in os.listdir(IMG_DIR):
        label_num = os.listdir(IMG_DIR).index(fold_list)

        fold_path = os.path.join(IMG_DIR,fold_list)
        fold = os.listdir(fold_path)

        for img in fold:
            img_array = cv2.imread(os.path.join(fold_path,img))

            img_128x128 = cv2.resize(img_array,((128,128)))
            img_array = img_128x128.flatten().reshape(-1,1).T
            labe_array = np.array([[label_num]])
            data_array  = np.concatenate(( labe_array,img_array), axis=1)

            if os.path.exists('nepenthes.csv'):
                with open('nepenthes.csv', 'ab') as f:
                    np.savetxt(f, data_array, delimiter=",",fmt='%f')
            else:
                with open('nepenthes.csv', 'w') as f:
                    columns = []
                    columns.append("label")
                    for i in range(1,data_array.shape[1]):   
                        columns.append("Pixel"+str(i))
                    columns = np.array(columns).reshape(1,-1)
                    np.savetxt(f, columns, delimiter=",",fmt='%s')
                    np.savetxt(f, data_array, delimiter=",",fmt='%f')

    train_csv = pd.read_csv('nepenthes.csv')
    train_data = train_csv.drop(train_csv.columns[[0]], axis=1)
    train_label = train_csv.iloc[:,0]
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, valid_idx = list(kf.split(train_data))[0]
    tr_data, val_data = train_data.iloc[train_idx], train_data.iloc[valid_idx]
    tr_label, val_label = train_label.iloc[train_idx], train_label.iloc[valid_idx]

    tr_data, val_data = np.array(tr_data/255.), np.array(val_data/255.)
    tr_data, val_data = tr_data.reshape(-1,128,128,3), val_data.reshape(-1,128,128,3)

    tr_label, val_label = to_categorical(tr_label,3), to_categorical(val_label,3)
    return tr_data, tr_label, val_data,  val_label



def create_model(tr_data, tr_label, val_data,  val_label):

    model = Sequential()

    model.add(Conv2D(filters = {{choice([32,64])}},
                     kernel_size = {{choice([(3,3),(5,5),(7,7)])}},
                     padding='same',
                     input_shape = (128,128,3)))
    model.add(Activation({{choice(["tanh", "relu"])}}))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = {{choice([32,64])}},
                     kernel_size = {{choice([(3,3),(5,5),(7,7)])}},
                     padding='same',
                     input_shape = (128,128,3)))
    model.add(Activation({{choice(["tanh", "relu"])}}))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout({{quniform(0.1, 0.5, 0.05)}}))

    model.add(Conv2D(filters = {{choice([32,64])}},
                     kernel_size = {{choice([(3,3),(5,5),(7,7)])}},
                     padding='same',
                     input_shape = (128,128,3)))
    model.add(Activation({{choice(["tanh", "relu"])}}))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = {{choice([32,64])}},
                     kernel_size = {{choice([(3,3),(5,5),(7,7)])}},
                     padding='same',
                     input_shape = (128,128,3)))
    model.add(Activation({{choice(["tanh", "relu"])}}))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout({{quniform(0.1, 0.5, 0.05)}})) 
        
    model.add(Flatten())

    if {{choice(['one', 'two'])}} == 'one':    
        model.add(Dense({{choice([500,600,700])}},
                        Activation({{choice(["tanh", "relu"])}})))
        model.add(BatchNormalization()) 
        model.add(Dropout({{quniform(0.1, 0.5, 0.05)}})) 


    elif {{choice(['one', 'two'])}} == 'two':        
        model.add(Dense({{choice([500,600,700])}},
                        Activation({{choice(["tanh", "relu"])}})))
        model.add(BatchNormalization()) 
        model.add(Dropout({{quniform(0.1, 0.5, 0.05)}})) 
        model.add(Dense({{choice([500,600,700])}},
                        Activation({{choice(["tanh", "relu"])}})))
        model.add(BatchNormalization()) 
        model.add(Dropout({{quniform(0.1, 0.5, 0.05)}}))    

    model.add(Dense(units = 3, activation = 'sigmoid'))

    model.compile(
        optimizer= {{choice(["adam",'rmsprop'])}}, 
        loss="binary_crossentropy",
        metrics=['binary_accuracy', 'Recall', 'Precision', 'AUC'] 
    )
    batch_size = {{choice([8,16,32])}}
    epochs = 100

    earlyStopping = callbacks.EarlyStopping(patience=20,monitor='val_loss', restore_best_weights=True)

    result = model.fit(
        tr_data,
        tr_label,
        validation_data = (val_data, val_label),
        batch_size=batch_size,
        epochs = epochs,
        callbacks = [earlyStopping],
        )
    model.summary()
    print(result)
    binary_accuracy = np.amax(result.history['binary_accuracy'])

    return {'loss': -binary_accuracy, 'status': STATUS_OK, 'model': model}



best_run, best_model = optim.minimize(model=create_model,
                                    data=prepare_data,
                                    algo=tpe.suggest,
                                    max_evals = 100,
                                    eval_space = True,
                                    trials = Trials(),
                                    )
_, _ ,val_data, val_label = prepare_data()
val_loss, val_acc, val_Recall, val_Precision, val_AUC = best_model.evaluate(val_data, val_label)
print("best_run:",best_run)
print("val_loss:",val_loss)
print("val_acc:",val_acc)

