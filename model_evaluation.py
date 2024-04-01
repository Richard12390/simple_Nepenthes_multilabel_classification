import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import requests
import urllib3
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MultiLabelBinarizer

with open("model_structure.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model_weights.h5")

train_dir = os.path.join("input", "train")
test_dir = os.path.join("input", "test")

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), class_mode='sparse')

custom_labels = ['truncata', 'veithcii', 'ventricosa']

test_data.class_indices = {label:index for index, label in enumerate(custom_labels)}

test_data_list, test_label_list = [], []

for _ in range(len(test_data)):
     batch_images , batch_labels  = test_data.next()
     batch_labels = [[custom_labels[int(label)]] for label in batch_labels]
     test_data_list.extend(batch_images)
     test_label_list.extend(batch_labels)

test_images = np.array(test_data_list)
test_labels = np.array(test_label_list)

mlb = MultiLabelBinarizer()
test_labels = mlb.fit_transform(test_labels)
 
predictions = model.predict(test_images)

threshold = []
for i in range(len(custom_labels)):
     fpr, tpr, thresholds = roc_curve(test_labels[:,i], predictions[:,i],drop_intermediate=False)
     print(thresholds)
     Youdens_J = tpr-fpr
     Youdens_J_index = np.argmax(Youdens_J)
     threshold.append(thresholds[Youdens_J_index])
print("threshold",threshold)


predictions_result = np.where(predictions>np.array(threshold),1,0)

urls=[
     "https://scontent-tpe1-1.xx.fbcdn.net/v/t31.18172-8/14409511_808260242650517_9074443558818745378_o.jpg?_nc_cat=101&ccb=1-7&_nc_sid=5f2048&_nc_ohc=ezUJ3exTbB0AX-mo9cv&_nc_ht=scontent-tpe1-1.xx&oh=00_AfBBNOkwfUZ-W7suh03zoY0i7mEoUEFw3nzXOkfw6_NxjA&oe=66193730",
     "https://images.fotop.net/albums6/averyorchids/truncataveitchii/nEO_IMG_DSC_8923_Nepenthes_truncata_x_veitchii.jpg"
       ]
urllib3.disable_warnings()

threshold_dict = {}
for i, url in enumerate(urls):

     plt.subplot(len(urls)//4+1,3,i+1)     
     r = requests.get(url, stream=True).raw
     image = np.asarray(bytearray(r.read()), dtype="uint8")
     image = cv2.imdecode(image, cv2.IMREAD_COLOR)

     prediction_resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
     prediction_resized_image = (prediction_resized_image / 255.0).reshape(-1, input_shape[0], input_shape[1], input_shape[2])

     prediction = model.predict(prediction_resized_image)
     prediction = zip(list(mlb.classes_), list(prediction[0]))
     predictions = sorted(prediction, key=lambda x: x[1], reverse=True)
     print("prediction", prediction)
     
     predictions_list=[]
     for prediction in predictions:
          if prediction[1] > threshold_dict[prediction[0]]:
               predictions_list.append(f"{prediction[0]}: {100*prediction[1]:.2f}%")

     print("predictions_list",predictions_list)
     title_string = "\n".join(predictions_list)

     plt.title(title_string)
     plt.imshow(image[:,:,::-1])

plt.tight_layout()       
plt.show()             


