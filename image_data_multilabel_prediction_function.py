import os
import cv2
import numpy as np
from tensorflow import keras
from keras.models import model_from_json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve
from keras.preprocessing.image import ImageDataGenerator

def image_data_multilabel_prediction(folder_path,species_group,model_structure,model_weights):

	with open(model_structure, "r") as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json)
	model.load_weights(model_weights)

	threshold_dict = {}

	test_dir = os.path.join("input", "test")
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_data = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), class_mode='sparse')

	test_data.class_indices = {label:index for index, label in enumerate(species_group)}
	test_data_list, test_label_list = [], []

	for _ in range(len(test_data)):
		batch_images , batch_labels  = test_data.next()
		batch_labels = [[species_group[int(label)]] for label in batch_labels]
		test_data_list.extend(batch_images)
		test_label_list.extend(batch_labels)

	test_images = np.array(test_data_list)
	test_labels = np.array(test_label_list)

	mlb = MultiLabelBinarizer()
	test_labels = mlb.fit_transform(test_labels)
	
	predictions = model.predict(test_images)

	threshold_dict = {}
	for i in range(len(species_group)):
		fpr, tpr, thresholds = roc_curve(test_labels[:,i], predictions[:,i],drop_intermediate=False)
		Youdens_J = tpr-fpr
		Youdens_J_index = np.argmax(Youdens_J)
		threshold_dict[species_group[i]]=(thresholds[Youdens_J_index])

	print("threshold_dict",threshold_dict)

	input_shape = (128, 128, 3)
	image_multilabel_prediction_dict = {}
	for image in os.listdir(folder_path):
		if image.lower().endswith(('.jpg', '.jpeg', '.png')):
			image_path = os.path.join(folder_path, image)
			try:
				image = cv2.imread(image_path)

				prediction_resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
				prediction_resized_image = (prediction_resized_image / 255.0).reshape(-1, input_shape[1], input_shape[0], input_shape[2])

				prediction = model.predict(prediction_resized_image)
				prediction = dict(zip(list(mlb.classes_), list(prediction[0])))

				image_multilabel_prediction_dict[image_path] = {}

				for label in species_group:
					if (label in prediction) and (prediction[label] > threshold_dict[label]):
						image_multilabel_prediction_dict[image_path][label] = "True"
					else:
						image_multilabel_prediction_dict[image_path][label] = "False"
				if all(prediction[label] <= threshold_dict[label] for label in species_group):
					os.remove(image_path)
					print(f"delete {image_path}")
			except Exception as error:
				print(f"Error processing image '{image_path}': {error}")

	image_data_multilabel_prediction_dict={}
	for file_path in image_multilabel_prediction_dict:
		fold_name, file_name = os.path.split(file_path)
		image_name = os.path.splitext(file_name)[0]
		species, serial, source = image_name.split('_')
		row = {"species":species, "serial":serial, "source":source}
		for label in species_group:
			if label in image_multilabel_prediction_dict[file_path]:
				row[label] = image_multilabel_prediction_dict[file_path][label]
		image_data_multilabel_prediction_dict[file_path] = row

	return image_data_multilabel_prediction_dict
