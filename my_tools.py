from PIL import Image
from multiprocessing import Pool
import numpy as np

def matrix_to_array_for_PIL(matrix, rotate=True):
	'''
	Vectorizez the matrix to a 1D array.
	'''
	vec = []
	row_num, col_num = len(matrix), len(matrix[0])

	if not rotate:
		for j in range(col_num):
			for i in range(row_num):
				vec.append(matrix[i][j])
		return vec

	for i in range(row_num):
		for j in range(col_num):
			vec.append(matrix[i][j])
	return vec

def read_image_into_matrix(name):
	img_1 = Image.open(name)
	px_img_1 = img_1.load()

	width, height = img_1.size

	matrix_img_1 = []
	for i in range(width):
		matrix_img_1.append([])
		for j in range(height):
			matrix_img_1[-1].append(px_img_1[i, j])

	return matrix_img_1

def get_image_dimensions(name):
	img_1 = Image.open(name)
	px_img_1 = img_1.load()
	return img_1.size

def write_image_from_matrix(name, matrix, flag=True):
	im= Image.new('RGB', (len(matrix), len(matrix[0])))
	im.putdata( matrix_to_array_for_PIL( matrix , flag ) )
	im.save(name)

def reduce_height(matrix, start_flag=True):
	row_num, col_num = len(matrix), len(matrix[0])
	px_img_1_lr = []
	discarded_matrix = []
	for i in range(row_num):
		line_flag = start_flag

		px_img_1_lr.append([])
		discarded_matrix.append([])

		for j in range(col_num):
			if line_flag:
				px_img_1_lr[-1].append(matrix[i][j])
			else:
				discarded_matrix[-1].append(matrix[i][j])
			line_flag = not line_flag

		start_flag = not start_flag

	return (px_img_1_lr, discarded_matrix,)

def agreggate_on_height(matrix1, matrix2, elem_acces_func, start_flag=True):
	row_num, col_num = len(matrix1), len(matrix1[0])

	if row_num != len(matrix2) or col_num != len(matrix2[0]):
		print('Matrices don\'t have the same dimensions !')
		return

	res_matrix = []

	col_num *= 2

	for i in range(row_num):
		line_flag = start_flag

		res_matrix.append([])

		ii = 0
		jj = 0
		for _ in range(col_num):
			if line_flag:
				res_matrix[-1].append(elem_acces_func(\
					matrix1,\
					i,\
					ii\
					))
				ii += 1
			else:
				res_matrix[-1].append(elem_acces_func(\
					matrix2,\
					i,\
					jj\
					))
				jj += 1
			line_flag = not line_flag

		start_flag = not start_flag

	return res_matrix

def reduce_width(matrix, start_flag=True):
	row_num, col_num = len(matrix), len(matrix[0])

	px_img_1_lr = [[] for _ in range(int(row_num/2))]
	discarded_matrix = [[] for _ in range(int(row_num/2))]

	for j in range(col_num):
		line_flag = start_flag

		ii = 0
		jj = 0

		for i in range(row_num):
			if line_flag:
				px_img_1_lr[ii].append(matrix[i][j])
				ii += 1
			else:
				discarded_matrix[jj].append(matrix[i][j])
				jj += 1


			line_flag = not line_flag

		start_flag = not start_flag


	# print(len(px_img_1_lr))
	# print(len(px_img_1_lr[0]))

	return (px_img_1_lr, discarded_matrix,)

def agreggate_on_width(matrix1, matrix2, elem_acces_func, start_flag=True):
	row_num, col_num = len(matrix1), len(matrix1[0])

	if row_num != len(matrix2) or col_num != len(matrix2[0]):
		print('Matrices don\'t have the same dimensions !')
		return

	row_num *=2

	res_matrix = [[] for _ in range(row_num)]

	for j in range(col_num):
		line_flag = start_flag

		ii = 0
		jj = 0

		for i in range(row_num):
			if line_flag:
				res_matrix[i].append(elem_acces_func(\
					matrix1,\
					ii,\
					j\
					))
				ii += 1
			else:
				res_matrix[i].append(elem_acces_func(\
					matrix2,\
					jj,\
					j\
					))
				jj += 1

			line_flag = not line_flag

		start_flag = not start_flag

	return res_matrix

def get_file_name(file_path):
	i = len(file_path) - 1
	while True:
		if file_path[i] == '/':
			break
		i -= 1
	return file_path[i+1:]

def padd_image_with_zeros(name, goal_width=1020, goal_height=1020):
	'''
	Rewrites image on disk with zero padding.
	'''
	image_width, image_height = get_image_dimensions(name)
	if image_width < goal_width or image_height < goal_height:
		image_matrix = read_image_into_matrix(name)
		padded_line = [(0,0,0,) for _ in range(goal_height)]
		padded_matrix = []
		for i in range(goal_width):
			if i >= image_width:
				padded_matrix.append(padded_line)
			else:
				padded_matrix.append([])
				for j in range(goal_height):
					if j >= image_height:
						padded_matrix[-1].append((0,0,0,))
					else:
						padded_matrix[-1].append(image_matrix[i][j])
		write_image_from_matrix(name, padded_matrix)

def padd_image_with_PD_values(name, goal_width=1020, goal_height=1020):
	'''
	Rewrites images on disk with padding values sampled from the same probability
	distribution of the original images in order to reduce networks bias.
	'''
	image_width, image_height = get_image_dimensions(name)
	if image_width < goal_width or image_height < goal_height:
		image_matrix = read_image_into_matrix(name)

		color_hist = {}
		for row in image_matrix:
			for el in row:
				if el not in color_hist:
					color_hist[el] = 1
				else:
					color_hist[el] += 1

		pixel_num = image_width * image_height
		elements = list(map(lambda el: el[0], color_hist.items()))
		elem_indices = range(len(elements))
		probabilities = list(map(lambda el: el[1]/pixel_num, color_hist.items()))

		if image_height < goal_height:
			diff = goal_height - image_height
			for row in image_matrix:
				row += list(map(lambda x: elements[x],
					np.random.choice(\
						elem_indices, diff, probabilities).tolist()))

		if image_width < goal_width:
			for _ in range(goal_width - image_width):
				image_matrix.append(\
					list(map(lambda x: elements[x],
					np.random.choice(\
						elem_indices, goal_height, probabilities).tolist())))
		write_image_from_matrix(name, image_matrix, False)

		print('image path: ' + name)

def multiprocess_task(function, arguments, p_num):
	'''
	Function that delegates work to processes.
	function - that takes a sub-list of elements from arguments
	arguments - a list of elements
	p_num - number of workers
	'''
	f_div = len(arguments) // p_num
	process_work = []
	i = 0
	for _ in range(p_num):
		process_work.append(arguments[i:i+f_div])
		i += f_div
	b = 0
	for j in range(i, len(arguments)):
		process_work[b].append(arguments[j])
		b = b+1
	with Pool(p_num) as p:
		return p.map(function, process_work)

def numpy_acces_elem(matrix, x, y):
	return tuple(map(lambda el: int(el), matrix[x][y].tolist()))

def color_array_to_tuple(matrix):
	return list(map(\
		lambda row: list(map(\
			lambda el: tuple(el), row\
			)),\
		matrix,
	))