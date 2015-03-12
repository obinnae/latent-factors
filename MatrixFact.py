import csv
import numpy as np
# TODO: update e cutoff, iterations, lambda, and ita values (they caused floating-point calculation RuntimeWarning)
# TODO: rename "user_movie_matrix"


RATINGS_FILE_PATH = 'data.txt'
MOVIES_FILE_PATH = 'movies.txt'
NUM_USERS = 943


# TODO: complete docstring
# TODO: rename "ita", "eij", and "error"
def matrix_factorization(user_movie_matrix, dimensions, ita=0.0002, iterations=5000, lambda_=0.02):
    """ ...

    :param user_movie_matrix: ...
    :param dimensions: ...
    :param ita: ...
    :param iterations: ...
    :param lambda_: ...
    :return: ...
    :rtype: tuple
    """
    m, n = user_movie_matrix.shape
    u = np.random.rand(m, dimensions)
    v = np.random.rand(dimensions, n)
    for iteration in xrange(iterations):
        for i_user in xrange(m):
            for j_movie in xrange(n):
                # Only calculate non-missing values
                if user_movie_matrix[i_user][j_movie] > 0:
                    eij = user_movie_matrix[i_user][j_movie] - np.dot(u[i_user, :], v[:, j_movie])
                    # Gradient descent
                    for dimension in xrange(dimensions):
                        u[i_user][dimension] -= ita * (lambda_ * u[i_user][dimension] - 1 * v[dimension][j_movie] * eij)
                        v[dimension][j_movie] -= ita * (lambda_ * v[dimension][j_movie] - 1 * u[i_user][dimension] * eij)
        u_dot_v = np.dot(u, v)
        error = 0
        for i_user in xrange(m):
            for j_movie in xrange(n):
                if user_movie_matrix[i_user][j_movie] > 0:
                    error += (user_movie_matrix[i_user][j_movie] - u_dot_v[i_user, j_movie]) ** 2
                    # Frobenius norm
                    for dimension in xrange(dimensions):
                        error += lambda_ / 2 * (u[i_user][dimension] ** 2 + v[dimension][j_movie] ** 2)
        if error < 0.001:
            break
    return u, v


# TODO: complete docstring
def read_data(ratings_file_path, movies_file_path):
    """ ...

    :param ratings_file_path: ...
    :param movies_file_path: ...
    :return: ...
    :rtype: numpy.array
    """
    with open(ratings_file_path, 'rU') as ratings_file:
        ratings_reader = csv.reader(ratings_file, dialect=csv.excel_tab)
        ratings = np.array([[int(x) for x in rating] for rating in ratings_reader])
    with open(movies_file_path, 'rU') as movie_file:
        movie_tags = np.array(list(csv.reader(movie_file, dialect=csv.excel_tab)))
    num_movies = movie_tags.shape[0]
    user_movie_matrix = np.zeros((NUM_USERS, num_movies), dtype=np.int8)
    for user_id, movie_id, rating in ratings:
        user_movie_matrix[user_id - 1, movie_id - 1] = rating
    return user_movie_matrix


def run():
    # user_movie_matrix = read_data(RATINGS_FILE_PATH, MOVIES_FILE_PATH)
    test_matrix = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    u, v = matrix_factorization(test_matrix, dimensions=2)
    print np.dot(u, v)


if __name__ == '__main__':
    run()
