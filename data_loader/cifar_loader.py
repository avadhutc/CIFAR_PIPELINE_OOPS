from base.data_loader_base import DataLoader
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical

class CifarLoader(DataLoader):
	def __init__(self, config):		
		super().__init__(config)
		print(1)
		return

	#dataset_path = "/content/CIFAR_OOPS_PPL/dataset/cifar10"	
	def create_dataset(self):
		data_path = "/content/CIFAR_OOPS_PPL/dataset/cifar10/"
		dataset_path = "/content/CIFAR_OOPS_PPL/dataset/cifar10/test"
		appended_data = []
		#for directory in os.listdir(self.config.config_namespace.dataset_path):
		for directory in os.listdir(dataset_path):
		  #print(directory)
		  #print(os.path.join(path, directory))
		  #dir_path = os.path.join(self.config.config_namespace.dataset_path, directory)
		  dir_path = os.path.join(dataset_path, directory)
		  if os.path.isdir(dir_path):
		    #print(1)
		    for filename in os.listdir(dir_path):
		      #print(dir_path + '/' + filename)
		      #data = pd.read_excel(filename)
		      data = {'path' : dir_path + '/' + filename, 'label': directory}
		      appended_data.append(data)
		#with open(self.config.config_namespace.dataset_path + 'dataset.pickle', 'wb') as handle:
		with open(data_path + '/' + 'dataset.pickle', 'wb') as handle:
		  pickle.dump(appended_data, handle)		
 		
		#return

	def split_dataset(self):
				
		data_path = "/content/CIFAR_OOPS_PPL/dataset/cifar10"
		test_split_random_state = 123
		val_split_random_state = 123
		test_size = 0.2
		val_size = 0.2
		#with open(self.config.config_namespace.dataset_path + 'dataset.pickle', 'rb') as handle:
		with open(data_path + '/' + 'dataset.pickle', 'rb') as handle:
			data = pickle.load(handle)

		X = pd.DataFrame.from_dict(data)
		encoder= preprocessing.LabelEncoder()

		transfomed_label = encoder.fit_transform(X['label'])
		X_trainm, X_test, y_trainm, y_test = train_test_split(X['path'], transfomed_label, 
                                                    test_size=test_size, 
                                                    random_state=test_split_random_state)
		#print(X_trainm.shape, X_test.shape, y_trainm.shape, y_test.shape)
		X_train, X_val, y_train, y_val = train_test_split(X_trainm, y_trainm, 
		                                                    test_size=val_size, 
		                                                    random_state=val_split_random_state)
		#print(X_train.shape, X_val.shape, X_test.shape y_train.shape, y_val.shape)
		split_data = {"x_train": X_train, "y_train": y_train, "x_val": X_val, "y_val": y_val,
					  "x_test": X_test, "y_test": y_test}
		#with open(self.config.config_namespace.dataset_path + 'split.pickle', 'wb') as handle:
		with open(data_path + '/' + 'split.pickle', 'wb') as handle:
			pickle.dump(split_data, handle)

		print("Train data size:", X_trainm.shape[0])
		print("Validation data size:", X_val.shape[0])
		print("Test data size:", X_test.shape[0])

				
		#return

	def load_dataset(self):
		data_path = "/content/CIFAR_OOPS_PPL/dataset/cifar10"
		#with open(self.config.config_namespace.dataset_path + 'split.pickle', 'rb') as handle:
		with open(data_path + '/' + 'split.pickle', 'rb') as handle:
			data = pickle.load(handle)

		##X = pd.DataFrame.from_dict(data,  orient='index')

		##print(X['x_train'])
		##print(X)
		# Read the training data images and thier labels from the disk.
		self.train_data = self.read_images(data['x_train'])
		self.train_labels = self.read_labels(data['y_train'])

		# Read the validation data images and thier labels from the disk.
		self.val_data = self.read_images(data['x_val'])
		self.val_labels = self.read_labels(data['y_val'])

		# Read the test data images and thier labels from the disk.
		self.test_data = self.read_images(data['x_test'])
		self.test_labels = self.read_labels(data['y_test'])

		return

	def read_images(self, path_to_data):
		
		X = []
		for img_name in path_to_data:
		  im = load_img(img_name, target_size=(self.config.config_namespace.image_width,
		  				self.config.config_namespace.image_height))
		  im = img_to_array(im)
		  X.append(im)
		  
		X = np.array(X)

		return X		

	def read_labels(self, path_to_labels):
		Y = []
		for label in path_to_labels:		  
		  Y.append(label)
		  
		Y = np.array(Y)

		return Y		

	def display_data_element(self, which_data, index):
		# Create a new figure.
		plt.figure()

		
		if(which_data == "train_data"):
			plt.imshow( self.train_data[index, : , : ] )
			#plt.imshow( self.trainData[index, : , : ] )
			# plt.show()
			plt.savefig('/content/CIFAR_OOPS_PPL/dataset/cifar10/sample_training.png', bbox_inches='tight')
			plt.close()

		elif(which_data == "test_data"):
			plt.imshow( self.test_data[index,:,:] )
			#plt.imshow( self.testData[index,:,:])
			# plt.show()
			plt.savefig('/content/CIFAR_OOPS_PPL/dataset/cifar10/sample_testing.png', bbox_inches='tight')
		else:
			print("Error: display_data_element: whicData parameter is invalid !")

		# Close the figure.
		plt.close()
		return

	def preprocessDataset(self):
		# Convert the integer pixel data to floating data to speed up keras execution.
		self.train_data = self.train_data.astype('float32')
		self.val_data = self.val_data.astype('float32')
		self.test_data = self.test_data.astype('float32')

		# Rescale the pixel values from orignal values to the values in range 0 10 1.
		self.train_data = self.train_data / self.config.config_namespace.image_pixel_size
		self.val_data = self.train_data / self.config.config_namespace.image_pixel_size
		self.test_data = self.test_data / self.config.config_namespace.image_pixel_size

		# Convert from categorical to  boolean one hot encoded vector.
		self.train_label_one_hot = to_categorical( self.train_labels)
		self.val_label_one_hot = to_categorical( self.val_labels)
		self.test_label_one_hot = to_categorical( self.test_labels )
		return
