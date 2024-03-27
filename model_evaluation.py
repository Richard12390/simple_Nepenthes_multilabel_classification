import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import requests
import urllib3
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

with open("model_structure.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model_weights.h5")

train_dir = os.path.join("input", "train")
test_dir = os.path.join("input", "test")

# def preprocess_img(IMG_DIR):
#      images=[]
#      labels=[] 
#      for fold_list in os.listdir(IMG_DIR):

#           fold_path = os.path.join(IMG_DIR,fold_list)
#           fold = os.listdir(fold_path)
#           for img in fold:    
#                img_array = cv2.imread(os.path.join(fold_path,img))
#                img_128x128 = cv2.resize(img_array,((128,128)))
#                images.append(img_128x128)
#                label = [fold_path.split("\\")[-1].split("_")[0]]
#                labels.append(label)
#                break
#      return images, labels
# train_data_list, train_label_list = preprocess_img(train_dir)
# train_data = np.array(train_data_list)/255.0
# train_labels = np.array(train_label_list)
# print("train_data.shape",train_data.shape)
# # print(train_data)
# print("train_labels.shape",train_labels.shape)
# print(train_labels)
# mlb = MultiLabelBinarizer()
# train_labels = mlb.fit_transform(train_labels)
# # print("test_images.shape",test_images.shape)
# print("train_labels",train_labels)
# # print("train_labels.classes",train_labels.classes_)
# print(mlb.classes_)
# print("train_labels.shape",train_labels.shape)


# batchsize=16
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# 設定訓練數據和驗證數據的路徑
# train_data = train_datagen.flow_from_directory(train_dir, target_size=(128, 128),class_mode='sparse')
# train_images, train_labels = train_data.next()
test_data = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), class_mode='sparse')
# test_images, test_labels = test_data.next()

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
print("predictions:" ,predictions)
# predictions_prob = predictions[:,1]
# print("predictions_prob",predictions_prob)





threshold = []
for i in range(len(custom_labels)):
     fpr, tpr, thresholds = roc_curve(test_labels[:,i], predictions[:,i],drop_intermediate=False)
     print(thresholds)
     Youdens_J = tpr-fpr
     Youdens_J_index = np.argmax(Youdens_J)
     threshold.append(thresholds[Youdens_J_index])
     # plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
  
     # plt.legend(loc='lower right')
     # plt.xlabel('False Positive Rate')
     # plt.ylabel('True Positive Rate')
     # plt.title('ROC Curve')
print("threshold",threshold)
# custom_colors = ["orange","blue","green"]
# fig, axes = plt.subplots(1,3,figsize=(15,5))
# for label_index,ax in enumerate(axes):
#      fpr, tpr, thresholds = roc_curve(test_labels[:, label_index], predictions[:, label_index])
#      Youdens_J_index = np.argmax(tpr-fpr)
#      roc_auc = auc(fpr, tpr)
#      ax.plot(fpr,tpr,marker=".",linestyle='--',color=custom_colors[label_index],
#              label=f'AUC for Label {custom_labels[label_index]} = {roc_auc:.2f}')
#      ax.scatter(fpr[Youdens_J_index], tpr[Youdens_J_index], marker='*', color="black", label="Best Threshold")
#      ax.set_xlabel('False Positive Rate')
#      ax.set_ylabel('True Positive Rate')
#      ax.set_title(f'ROC Curve for N. {custom_labels[label_index]}')
#      ax.legend(loc='lower right')

# plt.tight_layout(rect=(2,2,1,1))
# plt.show()





predictions_result = np.where(predictions>np.array(threshold),1,0)
# print("predictions_result",predictions_result)

# multilabel_cf_matrix = multilabel_confusion_matrix(test_labels,predictions_result)

# fig = plt.figure(figsize=(15,5))
# for i ,(label,matrix) in enumerate(zip(mlb.classes_,multilabel_cf_matrix)):
#      plt.subplot(1,3,i+1)
#      labels = [label,f'not_{label}']
#      sns.heatmap(matrix, annot = True, square = True, fmt = 'd', cbar = False, cmap = 'Blues', 
#                     xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)     
#      plt.title(labels[0])
# plt.tight_layout()
# plt.show()



# cf_matrix = confusion_matrix(test_labels.argmax(axis=1), predictions_result.argmax(axis=1),normalize='true')
# # plt.figure(figsize=(8, 6))
# sns.heatmap(cf_matrix, annot=True, fmt='.2f', cmap='Blues',
#              xticklabels=mlb.classes_, yticklabels=mlb.classes_)
# plt.xlabel('Predicted Labels',fontsize=12)
# plt.ylabel('True Labels',fontsize=12)
# plt.title('Confusion Matrix',fontsize=12)
# plt.show()


urls=[
     "https://scontent-tpe1-1.xx.fbcdn.net/v/t31.18172-8/14409511_808260242650517_9074443558818745378_o.jpg?_nc_cat=101&ccb=1-7&_nc_sid=5f2048&_nc_ohc=ezUJ3exTbB0AX-mo9cv&_nc_ht=scontent-tpe1-1.xx&oh=00_AfBBNOkwfUZ-W7suh03zoY0i7mEoUEFw3nzXOkfw6_NxjA&oe=66193730",
     "https://images.fotop.net/albums6/averyorchids/truncataveitchii/nEO_IMG_DSC_8923_Nepenthes_truncata_x_veitchii.jpg"
       ]
urllib3.disable_warnings()

threshold_dict = {}
for i,threshold in enumerate(threshold):
     key = custom_labels[i]
     threshold_dict[key] = threshold
print("threshold_dict",threshold_dict)
fig = plt.figure(figsize = (15, 15))
input_shape = (128, 128, 3)
for i, url in enumerate(urls):
     # if url.endswith('.jpg') or url.endswith('.jpeg') or url.endswith('.png'):
     
     plt.subplot(len(urls)//4+1,3,i+1)     # print("requests",requests.get(url))
     r = requests.get(url, stream=True).raw
     image = np.asarray(bytearray(r.read()), dtype="uint8")
     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
     # image = np.array(Image.open(r).convert('RGB'))

     prediction_resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
     prediction_resized_image = (prediction_resized_image / 255.0).reshape(-1, input_shape[1], input_shape[0], input_shape[2])

     prediction = model.predict(prediction_resized_image)

     prediction = zip(list(mlb.classes_), list(prediction[0]))
     # print(type(mlb.classes_.shape))
     # print(mlb.classes_)
     # print(type(prediction))
     prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
     print("prediction", prediction)
     for prediction in prediction:
          if prediction[1] > threshold_dict[prediction[0]]:
               plt.title(f"{prediction[0]}: {100*prediction[1]:.2f}%")
     plt.imshow(image[:,:,::-1])

     # plt.imshow(image[:,:,::-1])
plt.tight_layout()       
plt.show()             



# plt.figure(figsize = (10,8))
# sns.heatmap(cf_matrix, annot=False, xticklabels = sorted(set(test_labels)), yticklabels = sorted(set(test_labels)),cbar=False)
# plt.title('Normalized Confusion Matrix')
# plt.xticks()
# plt.yticks()
# plt.show()


    

# train_x, test_x, train_y, test_y = train_test_split(df['Filepath'], df['Labels'], test_size = 0.2, stratify = train_df['Labels'], shuffle = True, random_state = 1)
# print("train_x:",train_x)
# print("test_x:",test_x)



# fpr, tpr, thresholds = roc_curve(train_labels[:,2], predictions[:,2])
# # ,drop_intermediate=False
# # fpr, tpr, thresholds = roc_curve(train_labels.ravel(), predictions.ravel(),drop_intermediate=False)
# print("fpr:",fpr)
# print("fpr.shape:",fpr.shape)
# print("tpr:",tpr)
# print("thresholds:",thresholds)