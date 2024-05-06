import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import seaborn as sns
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

custom_labels = ['truncata', 'veithcii', 'ventricosa']
custom_colors = ["orange","blue","green"]

folder_path = "Nepenthes_web_images"
images=[os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
print(images)

thresholds = []
threshold_dict = {}

def plot_accuracy_and_loss(result_history):
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
     plt.show()

def plot_AUROC(model_structure,model_weights):
     with open(model_structure, "r") as json_file:
          model_json = json_file.read()
     model = model_from_json(model_json)
     model.load_weights(model_weights)
     test_dir = os.path.join("input", "test")
     test_datagen = ImageDataGenerator(rescale=1./255)
     test_data = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), class_mode='sparse')
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

     for i in range(len(custom_labels)):
          fpr, tpr, threshold = roc_curve(test_labels[:,i], predictions[:,i],drop_intermediate=False)
          Youdens_J = tpr-fpr
          Youdens_J_index = np.argmax(Youdens_J)
          thresholds.append(threshold[Youdens_J_index])

     fig, axes = plt.subplots(1,3,figsize=(15,5))
     for label_index,ax in enumerate(axes):
          fpr, tpr, threshold = roc_curve(test_labels[:, label_index], predictions[:, label_index])
          Youdens_J_index = np.argmax(tpr-fpr)
          roc_auc = auc(fpr, tpr)
          ax.plot(fpr,tpr,marker=".",linestyle='--',color=custom_colors[label_index],
               label=f'AUC for Label {custom_labels[label_index]} = {roc_auc:.2f}')
          ax.scatter(fpr[Youdens_J_index], tpr[Youdens_J_index], marker='*', color="black", label="Best Threshold")
          ax.set_xlabel('False Positive Rate')
          ax.set_ylabel('True Positive Rate')
          ax.set_title(f'ROC Curve for N. {custom_labels[label_index]}')
          ax.legend(loc='lower right')
     plt.tight_layout(rect=(2,2,1,1))
     plt.show()
     return model, mlb, predictions, test_labels

def plot_confusion_matrix(mlb, predictions, test_labels):
     predictions_result = np.where(predictions>np.array(thresholds),1,0)
     multilabel_cf_matrix = multilabel_confusion_matrix(test_labels,predictions_result)

     fig = plt.figure(figsize=(15,5))
     for i ,(label,matrix) in enumerate(zip(mlb.classes_,multilabel_cf_matrix)):
          plt.subplot(1,3,i+1)
          labels = [label,f'not_{label}']
          sns.heatmap(matrix, annot = True, square = True, fmt = 'd', cbar = False, cmap = 'Blues', 
                         xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)     
          plt.title(labels[0])
     plt.tight_layout()
     plt.show()

     cf_matrix = confusion_matrix(test_labels.argmax(axis=1), predictions_result.argmax(axis=1),normalize='true')
     sns.heatmap(cf_matrix, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=mlb.classes_, yticklabels=mlb.classes_)
     plt.xlabel('Predicted Labels',fontsize=12)
     plt.ylabel('True Labels',fontsize=12)
     plt.title('Confusion Matrix',fontsize=12)
     plt.show()

def image_prediction(model, mlb):
     for i,threshold in enumerate(thresholds):
          key = custom_labels[i]
          threshold_dict[key] = threshold
     fig = plt.figure(figsize = (10, 10))
     input_shape = (128, 128, 3)

     threshold_title = {key: f"{value*100:.2f}" for key, value in threshold_dict.items()}
     threshold_title = ", ".join([f"{key} = {value}%" for key, value in threshold_title.items()])

     plt.suptitle(f"Threshold: {threshold_title}")
     for i, image_path in enumerate(images):

          plt.subplot(len(images)//4+1,3,i+1)     
          image = cv2.imread(image_path)

          prediction_resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
          prediction_resized_image = (prediction_resized_image / 255.0).reshape(-1, input_shape[0], input_shape[1], input_shape[2])

          prediction = model.predict(prediction_resized_image)
          prediction = zip(list(mlb.classes_), list(prediction[0]))
          predictions = sorted(prediction, key=lambda x: x[1], reverse=True)
          
          predictions_list=[]
          for prediction in predictions:
               if prediction[1] > threshold_dict[prediction[0]]:
                    predictions_list.append(f"{prediction[0]}: {100*prediction[1]:.2f}%")

          title_string = "\n".join(predictions_list)

          plt.title(title_string)
          plt.imshow(image[:,:,::-1])

     plt.tight_layout()       
     plt.show()             

