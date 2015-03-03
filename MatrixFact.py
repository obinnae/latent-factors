import csv
import numpy as np 
import matplotlib.pyplot as plt 

f_data = 'data.txt'
f_movie = 'movies.txt'


with open(f_data, 'rU') as fin:
	ratings = np.array(list(csv.reader(fin, dialect=csv.excel_tab)))

#with open(f_movie, 'r') as fin:
#	movie_tag = np.array(list(csv.reader(fin)))

# Create a user-movie matrix Y, 0 represent missing values
#nRatings = ratings.shape[0]
nMovies = 1682
nUsers = 943

Y = np.zeros((nUsers,nMovies),dtype = np.int8)
for line in ratings:
	line = map(int,line)
	Y[line[0]-1, line[1]-1] = int(line[2])

#np.savetxt('miniproject2_data/rating_matrix.txt',Y,delimiter=',')
def matrix_factorization(Mat,k,iterations=5000,lambd=0.02,ita=0.0002):
	M, N = Mat.shape
	U = np.random.rand(M,k)
	V = np.random.rand(k,N)
	for iteration in xrange(iterations):
		for i in xrange(M):
			for j in xrange(N):
				# only calculate non-missing values
				if Mat[i][j] > 0:
					eij = Mat[i][j] - np.dot(U[i,:],V[:,j])
					for ik in xrange(k):
						# gradient descent
						U[i][ik] = U[i][ik] - ita*(lambd*U[i][ik]-1*V[ik][j]*eij)
						V[ik][j] = V[ik][j] - ita*(lambd*V[ik][j]-1*U[i][ik]*eij)

		error_Matrix = np.dot(U,V)
		# calculate labmda/2*(U^2+V^2)+sum((yij-np.dot(ui*vj))^2)
		e = 0
		for i in xrange(M):
			for j in xrange(N):
				if Mat[i][j] > 0:
					e = e + pow(Mat[i][j]-np.dot(U[i,:],V[:,j]), 2)
					# Frobenius norm
					for ik in xrange(k):
						e = e + lambd/2*(pow(U[i][ik],2)+pow(V[ik][j],2))

		if e < 0.001:
			break

	return U, V

if __name__ == '__main__':
	# R is small matrix for testing
	# if the matrix is large as Y, the algorithms works slowly, we need to cut down iterations parameter
	R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

 	R = np.array(R)
 	# 2 means 2 dimenstions.
	U, V = matrix_factorization(R,2)
	print np.dot(U,V)


