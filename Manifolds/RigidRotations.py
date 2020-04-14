import scipy
import numpy as np
from numpy import linalg as la

from pyquaternion import Quaternion
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation

from Tools.KarcherMean import KarcherMean
from Tools.Visualization import RotationVisualizer

from .AbstractManifold import AbstractManifold

SPECIAL_TOLERANCE = 0.001
ORTHOGONAL_TOLERANCE = 0.001


class RigidRotations(AbstractManifold):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

    def zero_func(self, x_0, x_1):
        return np.eye(self.dim)

    def distance(self, x, y):
        x_quat = self._quaternion_from_matrix(x)
        y_quat = self._quaternion_from_matrix(y)
        return Quaternion.distance(x_quat, y_quat)

    def _quaternion_from_matrix(self, matrix):
        """
        :param matrix: Rotation matrix.
        :return: Quaternion representation of the given matrix.
        """
        return Quaternion(Rotation.from_matrix(matrix).as_quat())

    def _matrix_from_quaternion(self, quaternion):
        """
        :param quaternion: Quaternion representation of a given rotation matrix.
        :return: The rotation matrix.
        """
        return Rotation.from_quat(quaternion.elements).as_matrix()

    def log(self, x, y):
        x_quat = self._quaternion_from_matrix(x)
        y_quat = self._quaternion_from_matrix(y)

        return np.array(list(Quaternion.log_map(x_quat, y_quat)))

    def exp(self, x, y):
        x_quat = self._quaternion_from_matrix(x)

        return self._matrix_from_quaternion(Quaternion.exp_map(x_quat, Quaternion(y)))

    def gen_point(self):
        matrix = np.array(special_ortho_group.rvs(self.dim))
        return matrix

    def zero_func(self, x_0, x_1):
        return np.eye(self.dim)

    def is_in_manifold(self, matrix):
        """
        Is in SO(3) predicate.
        :param matrix: Examined matrix.
        :return: bool
        """
        is_orthogonal = (la.norm(np.matmul(matrix, np.transpose(matrix)) - np.eye(self.dim)) < ORTHOGONAL_TOLERANCE)
        is_special = np.abs(la.det(matrix) - 1) < SPECIAL_TOLERANCE
        answer = is_special and is_orthogonal
        if not answer:
            print(matrix, is_special, is_orthogonal)
        return answer

    def geodesic_l2_mean_step(self, current_estimator, noisy_samples, weights, tolerance=0.00000001):
        projected_diff_samples = [w_i * scipy.linalg.logm(np.matmul(np.linalg.inv(current_estimator), noisy_sample)) 
                                  for w_i, noisy_sample in
                                  zip(weights, noisy_samples)]
        matrices_sum = np.zeros(projected_diff_samples[0].shape, dtype='complex128')
        for projected_diff_matrix in projected_diff_samples:
            matrices_sum += projected_diff_matrix
        r = matrices_sum / sum(weights)
        projected_avg_norm = np.linalg.norm(r, ord=2)
        if projected_avg_norm < tolerance:
            state_of_convergence = True
            return current_estimator, state_of_convergence
        else:
            new_estimator = np.matmul(current_estimator, scipy.linalg.expm(r))
            state_of_convergence = False
            return new_estimator, state_of_convergence

    def geodesic_l2_mean(self, noisy_samples, weights, tolerance=0.00000001, maximum_iteration=10):
        mean_estimator = noisy_samples[0]
        mean_estimator_list = [mean_estimator]
        state_of_convergence = False
        run_index = 0
        while not state_of_convergence and run_index < maximum_iteration:
            mean_estimator, state_of_convergence = self.geodesic_l2_mean_step(
                mean_estimator, noisy_samples, weights, tolerance)
            mean_estimator_list.append(mean_estimator)
            run_index += 1
        return mean_estimator_list

    def average(self, values_to_average, weights, base=np.eye(3)):
        return self.geodesic_l2_mean(values_to_average, weights)[-1]

    def _to_numbers(self, x):
        return self.distance(x, np.eye(3))

    def plot(self, data, title, filename, norm_visualization=False):
        if norm_visualization:
            return super().plot(data, title, filename)
        centers = np.zeros_like(data, dtype=object)
        for index in np.ndindex(data.shape):
            centers[index] = np.array([index[0], index[1], 0])
        print("start to visualize")
        RotationVisualizer(data, centers).save(filename, title)


def main():
    m = RigidRotations()

    a = m.gen_point()
    b = m.gen_point()
    c = m.gen_point()

    d = m.average([a,b,c], [1,1,1])
    return a,b,c,d,m


if __name__ == '__main__':
    a,b,c,d,m = main()
