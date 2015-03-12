import csv
import numpy as np
# TODO: update e cutoff, iterations, lambda, and ita values (they caused floating-point calculation RuntimeWarning)


RATINGS_FILE_PATH = 'data.txt'
MOVIES_FILE_PATH = 'movies.txt'
NUM_USERS = 943


# TODO: complete docstring
def matrix_factorization(user_movie_matrix, dimensions, iterations=5000, lambd=0.02, ita=0.0002):
    """ ...

    :param user_movie_matrix: ...
    :param dimensions: ...
    :param iterations: ...
    :param lambd: ...
    :param ita: ...
    :return: ...
    :rtype: ...
    """
    m, n = user_movie_matrix.shape
    u = np.random.rand(m, dimensions)
    v = np.random.rand(dimensions, n)
    for iteration in xrange(iterations):
        for user_index in xrange(m):
            for movie_index in xrange(n):
                # only calculate non-missing values
                if user_movie_matrix[user_index][movie_index] > 0:
                    eij = user_movie_matrix[user_index][movie_index] - np.dot(u[user_index, :], v[:, movie_index])
                    for ik in xrange(dimensions):
                        # gradient descent
                        u[user_index][ik] -= ita * (lambd * u[user_index][ik] - 1 * v[ik][movie_index] * eij)
                        v[ik][movie_index] -= ita * (lambd * v[ik][movie_index] - 1 * u[user_index][ik] * eij)
        error_matrix = np.dot(u, v)
        # calculate lambd/2*(u^2+v^2)+sum((yij-np.dot(ui*vj))^2)
        e = 0
        for user_index in xrange(m):
            for movie_index in xrange(n):
                if user_movie_matrix[user_index][movie_index] > 0:
                    user_movie_rating = user_movie_matrix[user_index][movie_index]
                    ui_dot_vj = np.dot(u[user_index, :], v[:, movie_index])
                    e += (user_movie_rating - ui_dot_vj) ** 2
                    # Frobenius norm
                    for ik in xrange(dimensions):
                        e += lambd / 2 * (u[user_index][ik] ** 2 + v[ik][movie_index] ** 2)
        if e < 0.001:
            break
    return u, v


# TODO: complete docstring
def read_data(ratings_file_path, movies_file_path):
    """ ...

    :param ratings_file_path: ...
    :param movies_file_path: ...
    :return: ...
    :rtype: ...
    """
    with open(ratings_file_path, 'rU') as data_file:
        data_reader = csv.reader(data_file, dialect=csv.excel_tab)
        ratings = np.array([[int(x) for x in line] for line in data_reader])
    with open(movies_file_path, 'rU') as movie_file:
        movie_tags = np.array(list(csv.reader(movie_file, dialect=csv.excel_tab)))
    num_movies = movie_tags.shape[0]
    user_movie_matrix = np.zeros((NUM_USERS, num_movies), dtype=np.int8)
    for line in ratings:
        user_movie_matrix[line[0] - 1, line[1] - 1] = line[2]
    # np.savetxt('miniproject2_data/rating_matrix.txt', user_movie_matrix, delimiter=',')
    return user_movie_matrix


def run():
    user_movie_matrix = read_data(RATINGS_FILE_PATH, MOVIES_FILE_PATH)
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
