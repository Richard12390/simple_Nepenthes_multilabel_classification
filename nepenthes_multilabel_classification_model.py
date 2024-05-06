#%%
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D,  BatchNormalization,Activation
from tensorflow.keras.models import Sequential
from package.model_evaluation import plot_AUROC, plot_confusion_matrix, image_prediction, plot_accuracy_and_loss

batch_size = 16
epochs = 1
model_structure = "model_structure.json"
model_weights = "model_weights.h5"

def model_build(batch_size, epochs, model_structure,model_weights):
     train_dir = os.path.join("input", "train")
     test_dir = os.path.join("input", "test")
     batch_size = batch_size
     epochs = epochs

     # image augmentation
     train_data = ImageDataGenerator(
          rescale=1.0 / 255,
          horizontal_flip=True,
          zoom_range=0.1,
          width_shift_range=0.1,
          height_shift_range=0.1,
          rotation_range=30,
          vertical_flip=False)
     validation_data = ImageDataGenerator(
          rescale=1.0 / 255,
          validation_split=0.2
     )
     test_data = ImageDataGenerator(
          rescale=1.0 / 255
     )
     train_images = train_data.flow_from_directory(
          directory=train_dir,
          target_size = (128,128),
          class_mode="categorical", batch_size=batch_size, shuffle=True, subset='training'
     )
     val_images = validation_data.flow_from_directory(
          directory=train_dir, 
          target_size = (128,128),
          class_mode="categorical", batch_size=batch_size, shuffle=True, subset='validation'
     )
     test_images = test_data.flow_from_directory(
          directory=test_dir,
          target_size = (128,128),
          class_mode="categorical", batch_size=batch_size, shuffle=False
     )

     labels = (train_images.class_indices)
     labels = dict((value,key) for key,value in labels.items())

     # model 
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
          metrics=[ 'AUC','accuracy','Recall', 'Precision',] )

     earlyStopping = EarlyStopping(patience=20,
                                   monitor='val_loss',
                                   restore_best_weights=True)

     modelcheckpoint = ModelCheckpoint( model_weights,
                                        monitor = 'val_auc',
                                        verbosesave_best_only = True)
     result = model.fit(
          train_images,
          validation_data = val_images,
          batch_size = batch_size,
          epochs = epochs,
          callbacks = [earlyStopping,modelcheckpoint],
          verbose = 1)
     model.summary()

     # save model
     model_json = model.to_json()
     with open(model_structure, "w") as json_file:
          json_file.write(model_json)

     result_history = result.history
     print(result_history)

     return model_structure, model_weights, result_history

if __name__ == '__main__':
     model_structure, model_weights, result_history = model_build(batch_size, epochs, model_structure,model_weights)
     plot_accuracy_and_loss(result_history)
     model, mlb, predictions, test_labels = plot_AUROC(model_structure, model_weights)
     plot_confusion_matrix(mlb, predictions, test_labels)
     image_prediction(model, mlb)

