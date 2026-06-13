import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

from base.bayesian import BaseBayesianClassifier


class QDA_Chol1(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.Ls = [
        cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True)
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L = self.Ls[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = solve_triangular(L, unbiased_x, lower=True)

    return -np.log(L.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        dtrtri(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True), lower=1)[0]
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()

class TensorizedChol(QDA_Chol3):
    '''
    Misma idea que TensorizedQDA pero sobre la version Cholesky: en vez de
    apilar las inversas de covarianza, apila las L^-1 (triangulares inferiores)
    en un tensor (k, p, p) y paraleliza el calculo sobre las k clases.
    Sigue prediciendo de a 1 observacion (el for sobre n vive en predict).
    '''

    def _fit_params(self, X, y):
        # QDA_Chol3 calcula self.L_invs (lista de k matrices (p,p)) y self.means
        super()._fit_params(X, y)

        # apilamos sobre un nuevo eje (las k clases) -> tensores
        self.tensor_L_inv = np.stack(self.L_invs)   # (k, p, p)
        self.tensor_means = np.stack(self.means)    # (k, p, 1)

    def _predict_log_conditionals(self, x):
        # x: (p, 1) -> broadcasting contra las k clases
        unbiased_x = x - self.tensor_means          # (k, p, 1)

        # y_j = L_j^-1 (x - mu_j) para las k clases de una (matmul batched)
        y = self.tensor_L_inv @ unbiased_x          # (k, p, p) @ (k, p, 1) = (k, p, 1)

        # log|L^-1| por clase = log(prod de su diagonal)
        log_det = np.log(np.diagonal(self.tensor_L_inv, axis1=1, axis2=2).prod(axis=1))  # (k,)

        # ||y_j||^2 sumando sobre el eje de features (p)
        return log_det - 0.5 * (y**2).sum(axis=(1, 2))  # (k,)

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class EfficientChol(TensorizedChol):
    '''
    Combina TensorizedChol (tensoriza sobre clases) con el insight de
    EfficientQDA (elimina el for sobre las n observaciones sin construir la
    matriz n x n). Como ||y_i||^2 = y_i^T y_i, la diagonal que necesitamos es
    directamente la suma de cuadrados de Y sobre el eje de features.
    '''

    def predict(self, X):
        # X: (p, n) -> broadcasting contra las k clases
        unbiased_X = X - self.tensor_means              # (k, p, n)

        # Y = L^-1 (X - mu) para las k clases y las n obs de una sola pasada
        Y = self.tensor_L_inv @ unbiased_X              # (k, p, p) @ (k, p, n) = (k, p, n)

        # ||y_i||^2 por clase y por obs = suma de cuadrados sobre el eje p
        # (sin armar la matriz n x n)
        inner_prod_diag = np.sum(Y**2, axis=1)          # (k, n)

        log_det = np.log(np.diagonal(self.tensor_L_inv, axis1=1, axis2=2).prod(axis=1))  # (k,)
        log_conditionals = log_det[:, None] - 0.5 * inner_prod_diag                       # (k, n)

        return np.argmax(self.log_a_priori[:, None] + log_conditionals, axis=0).reshape(1, -1)

