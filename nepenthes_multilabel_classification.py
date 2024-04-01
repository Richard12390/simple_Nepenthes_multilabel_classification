
#%%
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()

# preprocessing images 
train_dir = os.path.join("input", "train")
test_dir = os.path.join("input", "test")

# image preprocessing 
batch_size = 16
epochs = 100

# image augmentation
train_data = ImageDataGenerator(
     rescale=1.0 / 255,
     horizontal_flip=True,
     zoom_range=0.1,
     width_shift_range=0.1,
     height_shift_range=0.1,
     rotation_range=30,
     vertical_flip=False,
)
train_images = train_data.flow_from_directory(
     directory=train_dir,
     target_size = (128,128),
     class_mode="binary", batch_size=batch_size, shuffle=True, subset='training' #training split from training data
)

labels = (train_images.class_indices)
labels = dict((value,key) for key,value in labels.items())

# model build
from sklearn.model_selection import KFold 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D,  BatchNormalization,Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 

model = Sequential()

model.add(Conv2D(filters = 64,
                    kernel_size = (7,7),
                    padding='same',
                    activation = "tanh",
                    input_shape = (128,128,3)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64,
                    kernel_size = (7,7),
                    padding='same',
                    activation = "relu",
                    input_shape = (128,128,3)))

model.add(BatchNormalization())  
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters = 64,
                    kernel_size = (7,7),
                    padding='same',
                    activation = "relu",
                    input_shape = (128,128,3)))

model.add(BatchNormalization())
model.add(Conv2D(filters = 64,
                    kernel_size = (3,3),
                    padding='same',
                    activation = "relu",
                    input_shape = (128,128,3)))

model.add(BatchNormalization())  
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4)) 
     
model.add(Flatten())

model.add(Dense(700,
               activation = "tanh"))
model.add(BatchNormalization()) 
model.add(Dropout(0.25)) 
model.add(Dense(700,
               activation = "tanh"))
model.add(BatchNormalization()) 
model.add(Dropout(0.4))    

model.add(Dense(units = 3, activation = 'sigmoid'))

model.compile(
    optimizer= 'rmsprop', 
    loss="binary_crossentropy",
    metrics=['accuracy', 'f1_score', 'average_precision'])

earlyStopping = EarlyStopping(  patience=2,
                                monitor='val_f1_score',
                                restore_best_weights=True)

modelcheckpoint = ModelCheckpoint( 'model_weights.h5',
                                    monitor = 'val_auc',
                                    verbosesave_best_only = True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, train_index, val_index in enumerate(kf.split(train_images), 1):
    X_train, X_val = train_images[train_index], train_images[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # train model
    result = model.fit(
        X_train,
        validation_data=X_val,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[earlyStopping, modelcheckpoint],
        verbose=1)
    fold_results.append(result.history)

for fold, result in enumerate(fold_results, 1):
    print(f"Fold {fold} - Accuracy: {result['accuracy'][-1]}, Loss: {result['loss'][-1]}, Val Accuracy: {result['val_accuracy'][-1]}, Val Loss: {result['val_loss'][-1]}")

model.summary()

model_json = model.to_json()
with open("model_structure.json", "w") as json_file:
    json_file.write(model_json)

result_history = result.history
print(result_history)

fig = plt.figure(figsize = (20, 7))
plt.subplot(121)
plt.plot(result_history['accuracy'], label = 'acc')
plt.plot(result_history['val_accuracy'], label = 'val_acc')
plt.grid()
plt.legend()

plt.subplot(122)
plt.plot(result_history['loss'], label = 'loss')
plt.plot(result_history['val_loss'], label = 'val_loss')
plt.grid()
plt.legend()

