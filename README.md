# DefiningImageOperations
 defining image operations such as product , sum, difference, etc. of two images, 

I have defined 6 different operations on images:

there are differet products:
1) A * B where A and B are RGB matrices (hadamard product)
2) C * B where C is scalar and B is RGB matrix (regular scalar and matrix product)

1) hadamard product
binary operator.
hadamard product: A * B , where A and B are RGB matrix.
In this context I've defined RGB vector multiplication as follows:
if you have two RGB vectors D and B, then D * B = cross product of given matrices.
To sum it up result of hadamard product is a matrix which has cross product of two input vectors as it's elements.

P.S. after defining this operation every time where product of two matrices is needed hadamard product is used two avoid
getting out of limit of RGB color values (RGB color values in my code are represented with floating point numbers)

2) inverse of an image
unary operator.
In this context I've defined inverse of an image as follows:
Because of having NxNx3 matrix with RGB vectors as elements, regular inverse of a matrix couldn't be defined.
To do that firstly i had to define inverse of a 1x3 RGB vector. 
inverse of a RGB vector:
If we have RGB vector A = [x, y, z], then A^-1 = [255 - x, 255 - y, 255 - z,]
after that inverse of a matrix is trivial: every element (a.k.a RGB vector) = [255 - x, 255 - y, 255 - z,] where (x, y, z)
is the original vector

3)image addition 4)image subtraction
binary operations
These operations work as originaly intended.
A ± B , where A and B are matrices outputs matrix whose elements are sum(difference) of corresponding vectors of A and B.


5-6) similarity transforamtion
It's done in two different ways:
• B = P^-1 * A * P where A is NxN RGB matrix and P is NxN scalar matrix
• B = P^-1 * A * P where A and P are NxN RGB matrices
