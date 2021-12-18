import numpy as np
import cv2 as cv

ROW = COLUMN = 600
np.random.seed(0)


class Vector:
    def __init__(self, nums):
        self.nums = nums

    def show_vector(self):
        print(self.nums)

    def __str__(self):
        return ", ".join(map(str, self.nums))

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return [a + b for a, b in zip(self.nums, other.nums)]

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return [round(other * num, 4) for num in self.nums]

        if isinstance(other, Vector):
            return np.cross(self.nums, other.nums)

    __rmul__ = __mul__

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return np.abs([a - b for a, b in zip(self.nums, other.nums)])


class Matrix:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def print_matrix(self):
        for line in self.matrix:
            print('  '.join(map(str, line)))
        print()

    def get_shape(self):
        return self.matrix.shape

    def get_inverse(self):
        white = Vector([255, 255, 255])
        tmp = np.array(self.matrix)
        for i in range(len(tmp)):
            for j in range(len(tmp[i])):
                tmp_vector = Vector(tmp[i][j])
                tmp[i][j] = white - tmp_vector
        return Matrix(tmp)

    def __str__(self):
        tmp = ''
        for row in self.matrix:
            tmp += (' '.join(map(str, row))) + '\n'
        return tmp

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix([[Vector(a) + Vector(b) for a, b in zip(c, d)] for c, d in zip(self.matrix, other.matrix)])

    def __mul__(self, other):
        if isinstance(other, list):
            return Matrix(np.abs([[Vector(a) * b for a, b in zip(c, d)] for c, d in zip(self.matrix, other)]))

        if isinstance(other, float) or isinstance(other, int):
            return Matrix(np.abs([[a * other for a in b] for b in self.matrix]))

        if isinstance(other, Matrix):
            return Matrix([[Vector(a) * Vector(b) for a, b in zip(c, d)] for c, d in zip(self.matrix, other.matrix)])

    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix([[Vector(a) - Vector(b) for a, b in zip(c, d)] for c, d in zip(self.matrix, other.matrix)])


def from_rgb_matrix_to_image(matrix):
    cv.imshow("transformation", matrix)
    cv.waitKey(0)


def visualise_inverse_of_image(image):
    img1_inverse = Matrix(np.array(image))
    matrix4 = img1_inverse.get_inverse()
    cv.imshow("inverse", matrix4.matrix)
    cv.waitKey(0)


def visualise_similarity_transformation_for_images_scalar_matrix(mat: Matrix, p, p_inverse):
    similar_matrix = ((p_inverse * mat) * p)
    k = np.linspace(0, 1, 11)
    for i in range(len(k)):
        result = ((k[i] * similar_matrix.matrix) + ((1 - k[i]) * mat.matrix))
        from_rgb_matrix_to_image(result)


def visualise_similarity_transformation_for_images_rgb_matrix(mat: Matrix, p, p_inverse):
    similar_matrix = ((p_inverse * mat) * p)
    k = np.linspace(0, 1, 11)
    for i in range(len(k)):
        result = (k[i] * similar_matrix.matrix + (1 - k[i]) * mat.matrix)
        from_rgb_matrix_to_image(result)


def visualise_hadamard_product(mat1: Matrix, mat2: Matrix):
    hadamard_product = (mat1 * mat2)
    cv.imshow("product", hadamard_product.matrix)
    cv.waitKey(0)


def visualise_image_addition(mat1: Matrix, mat2: Matrix):
    matrix_sum = (mat1 + mat2)
    cv.imshow("sum", matrix_sum.matrix)
    cv.waitKey(0)


def visualise_image_subtraction(mat1: Matrix, mat2: Matrix):
    matrix_subtract = (mat1 - mat2)
    cv.imshow("subtraction", matrix_subtract.matrix)
    cv.waitKey(0)


img1 = cv.imread("image.jpg")
img2 = cv.imread('image2.jpg')

img1 = cv.resize(img1, (COLUMN, ROW))
img2 = cv.resize(img2, (COLUMN, ROW))

matrix1 = Matrix(np.array(img1) / 500)
matrix2 = Matrix(np.array(img2) / 500)

sml_mat_scl = list(np.round(np.random.uniform(0, 1, (COLUMN, ROW)), 2))

sml_mat_scl_inv = list(np.linalg.inv(sml_mat_scl))
sml_mat_vec = Matrix(np.round(np.random.uniform(0, 1, (COLUMN, ROW, 3)), 2))
sml_mat_vec_inv = sml_mat_vec.get_inverse()

# visualise_similarity_transformation_for_images_scalar_matrix(matrix1, sml_mat_scl, sml_mat_scl_inv)
# visualise_similarity_transformation_for_images_rgb_matrix(matrix1, sml_mat_vec, sml_mat_vec_inv)
# visualise_hadamard_product(matrix1, matrix2)
# visualise_inverse_of_image(img1)
# visualise_image_addition(matrix1, matrix2)
# visualise_image_subtraction(matrix1, matrix2)

