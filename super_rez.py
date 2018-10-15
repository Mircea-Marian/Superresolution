from my_tools import *
import glob
from multiprocessing import Pool

from keras.models import Sequential
import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
from multiprocessing import Process, Pipe
import numpy as np
import pickle

X2_IMAGE_SET = '/home/mircea/Desktop'\
	+ '/proiecte/tensorflow_projs/data_sets/DIV2K_train_LR_bicubic/X2/'
IMG_1 = '0001x2.png'
def main1():
	result = reduce_height(\
		reduce_width(\
		read_image_into_matrix(X2_IMAGE_SET + IMG_1)))

	write_image_from_matrix('low_rez.png', result)


HR_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/hr_rez/'
LR_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/lr_rez/'
FIRST_OUTPUT_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/first_cnn_output/'
SECOND_OUTPUT_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/second_cnn_output/'
THIRD_OUTPUT_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/third_cnn_output/'
FIN_OUTPUTS_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/final_outputs/'
FIRST_CNN_RES_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/first_cnn_res/'
SECOND_CNN_RES_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/second_cnn_res/'
THIRD_CNN_RES_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/super_rez_510/third_cnn_res/'
ORIGINAL_PATH = '/home/mircea/Desktop/proiecte/tensorflow_projs/data_sets/DIV2K_train_LR_bicubic/X2/'

def padd_img_list(name_list):
	for name in name_list:
		# padd_image_with_zeros(name)
		padd_image_with_PD_values(name)
def main_add_padding():
	'''
	Main for padding pre-processing.
	'''
	a = glob.glob(HR_PATH + '*.png')
	np.random.seed(7)
	multiprocess_task(padd_img_list, a, 8)

def split_images(name_list):
	for name in name_list:
		original_matrix = read_image_into_matrix(HR_PATH + name)

		first_wide_matrix, second_wide_matrix = reduce_height(original_matrix)

		lr_matrix, first_matrix = reduce_width(first_wide_matrix)

		write_image_from_matrix(LR_PATH + name, lr_matrix, False)

		write_image_from_matrix(FIRST_OUTPUT_PATH + name, first_matrix, False)

		second_matrix, third_matrix = reduce_width(second_wide_matrix)

		write_image_from_matrix(SECOND_OUTPUT_PATH + name, second_matrix, False)

		write_image_from_matrix(THIRD_OUTPUT_PATH + name, third_matrix, False)
def main_process_data():
	'''
	Splits the 510x510 images into 4 255x255 images.
	'''
	a = list(\
		map(lambda el: get_file_name(el),\
			glob.glob(HR_PATH + '*.png')))
	multiprocess_task(split_images, a, 7)

def main_test_dimensions():
	a = list(\
		map(lambda el: get_file_name(el),\
			glob.glob(HR_PATH + '*.png')))
	b = list(map(\
			lambda name: get_image_dimensions(FIRST_OUTPUT_PATH + name),\
			a\
		))
	print('max width: ' + str(max( b, key=lambda aa: aa[0] )[0]))
	print('max height: ' + str(max( b, key=lambda aa: aa[1] )[1]))
	print('min width: ' + str(min( b, key=lambda aa: aa[0] )[0]))
	print('min height: ' + str(min( b, key=lambda aa: aa[1] )[1]))

def get_model(square_dim=255):
	'''
	Construct the simple CNN architecture which outputs 3 feature maps
	with values in the interval [0;255].
	'''
	np.random.seed(0)
	model = Sequential()
	model.add(keras.layers.Conv2D(filters=11, kernel_size=5, padding='same',\
		activation='relu', input_shape=(square_dim,square_dim,3)))
	model.add(keras.layers.Conv2D(filters=3, kernel_size=5, padding='same',\
		activation='sigmoid'))
	model.add(keras.layers.Lambda(lambda x: 255 * x))
	# model.summary()
	# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

class MyData:
	def __init__(self, percentage_for_train,\
		percentage_for_validation):
		a = list(\
			map(lambda el: get_file_name(el),\
				glob.glob(HR_PATH + '*.png')))
		total_num = len(a)
		num_for_train = int(percentage_for_train * total_num)
		num_for_validation = int(percentage_for_validation * total_num)
		if num_for_train + num_for_validation > total_num:
			print('Percentages sum over 1.0 !!!!')
			return

		self.train_lr_list = []
		self.train_first_list = []
		self.train_second_list = []
		self.train_third_list = []
		for name in a[:num_for_train]:
			self.train_lr_list.append(img_to_array(Image.open(LR_PATH + name)))
			self.train_first_list.append(img_to_array(Image.open(FIRST_OUTPUT_PATH + name)))
			self.train_second_list.append(img_to_array(Image.open(SECOND_OUTPUT_PATH + name)))
			self.train_third_list.append(img_to_array(Image.open(THIRD_OUTPUT_PATH + name)))
		self.train_lr_list = np.array(self.train_lr_list, dtype="float")
		self.train_first_list = np.array(self.train_first_list, dtype="float")
		self.train_second_list = np.array(self.train_second_list, dtype="float")
		self.train_third_list = np.array(self.train_third_list, dtype="float")

		self.validation_lr_list = []
		self.validation_first_list = []
		self.validation_second_list = []
		self.validation_third_list = []
		for name in a[num_for_train : num_for_train + num_for_validation]:
			self.validation_lr_list.append(img_to_array(Image.open(LR_PATH + name)))
			self.validation_first_list.append(img_to_array(Image.open(FIRST_OUTPUT_PATH + name)))
			self.validation_second_list.append(img_to_array(Image.open(SECOND_OUTPUT_PATH + name)))
			self.validation_third_list.append(img_to_array(Image.open(THIRD_OUTPUT_PATH + name)))
		self.validation_lr_list = np.array(self.validation_lr_list, dtype="float")
		self.validation_first_list = np.array(self.validation_first_list, dtype="float")
		self.validation_second_list = np.array(self.validation_second_list, dtype="float")
		self.validation_third_list = np.array(self.validation_third_list, dtype="float")

CREATE_MODEL = True
LOAD_MODEL = False
def main_train():
	'''
	Main for training a new model or loading a pretrained one
	and trining it some more.
	'''
	model_aquisition = LOAD_MODEL

	data = MyData(0.5, 0.25)

	if model_aquisition == LOAD_MODEL:
		first_model = keras.models.load_model('first_model.h5')
	else:
		first_model = get_model()
	first_model.fit(
		data.train_lr_list,
		data.train_first_list,
		validation_data=(\
			data.validation_lr_list,\
			data.validation_first_list,\
		),
		batch_size=16,
		epochs=5)
	first_model.save('first_model.h5')
	print('Finished with first_model !\n')

	if model_aquisition == LOAD_MODEL:
		second_model = keras.models.load_model('second_model.h5')
	else:
		second_model = get_model()
	second_model.fit(
		data.train_lr_list,
		data.train_second_list,
		validation_data=(\
			data.validation_lr_list,\
			data.validation_second_list,\
		),
		batch_size=16,
		epochs=5)
	second_model.save('second_model.h5')
	print('Finished with second_model !\n')


	if model_aquisition == LOAD_MODEL:
		third_model = keras.models.load_model('third_model.h5')
	else:
		third_model = get_model()
	third_model.fit(
		data.train_lr_list,
		data.train_third_list,
		validation_data=(\
			data.validation_lr_list,\
			data.validation_third_list,\
		),
		batch_size=16,
		epochs=5)
	third_model.save('third_model.h5')
	print('Finished with third_model !\n')

def agregate_res_list(tuple_list):
	float_to_int_vec = np.vectorize(lambda el: int(el))
	for tup in tuple_list:
		r1_matrix = agreggate_on_width(\
			float_to_int_vec(tup[0]),\
			float_to_int_vec(tup[1]),\
			numpy_acces_elem)
		r2_matrix = agreggate_on_width(\
			float_to_int_vec(tup[2]),\
			float_to_int_vec(tup[3]),\
			numpy_acces_elem)

		write_image_from_matrix(FIRST_CNN_RES_PATH + tup[4],\
			color_array_to_tuple(\
			float_to_int_vec(tup[1]).tolist()))

		write_image_from_matrix(SECOND_CNN_RES_PATH + tup[4],\
			color_array_to_tuple(\
			float_to_int_vec(tup[2]).tolist()))

		write_image_from_matrix(THIRD_CNN_RES_PATH + tup[4],\
			color_array_to_tuple(\
			float_to_int_vec(tup[3]).tolist()))

		write_image_from_matrix(FIN_OUTPUTS_PATH + tup[4],\
			agreggate_on_height(r1_matrix, r2_matrix,\
				lambda m, x, y: m[x][y]))
def main_test():
	'''
	Main for testing the architecture. Agregates the results
	from the three CNNs.
	'''
	first_model = keras.models.load_model('first_model.h5')
	second_model = keras.models.load_model('second_model.h5')
	third_model = keras.models.load_model('third_model.h5')

	a = list(\
		map(lambda el: get_file_name(el),\
			glob.glob(HR_PATH + '*.png')))

	percentage_for_train = 0.5
	percentage_for_validation = 0.25
	percentage_for_test = 0.25
	total_num = len(a)

	num_for_train = int(percentage_for_train * total_num)
	num_for_validation = int(percentage_for_validation * total_num)
	num_for_test = int(percentage_for_test * total_num)

	a = a[num_for_train + num_for_validation:num_for_train + num_for_validation\
	+ num_for_test]

	test_lr_list = []
	for name in a:
		test_lr_list.append(img_to_array(Image.open(LR_PATH + name)))

	test_lr_list = np.array(test_lr_list, dtype="float")
	first_output = first_model.predict(test_lr_list, batch_size=16)
	second_output = second_model.predict(test_lr_list, batch_size=16)
	third_output = third_model.predict(test_lr_list, batch_size=16)
	pickle.dump(\
		(test_lr_list,\
		first_output,\
		second_output,\
		third_output,\
		), open( "predicted_data.p", "wb" ) )

	# test_lr_list, first_output, second_output, third_output =\
	# 	pickle.load( open( "predicted_data.p", "rb" ) )

	print('Finished predicting !')

	work = []
	for i in range(num_for_test):
		work.append((\
			test_lr_list[i],\
			first_output[i],\
			second_output[i],\
			third_output[i],\
			a[i],\
		))
	multiprocess_task(agregate_res_list, work, 8)

def resize_img_list(name_list):
	for name in name_list:
		img = Image.open(name)
		img = img.resize((510, 510,))
		img.save(name)
def main_resize():
	a = glob.glob(HR_PATH + '*.png')
	multiprocess_task(resize_img_list, a, 8)

if __name__ == "__main__":
	main_train()
	main_test()
