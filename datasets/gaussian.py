import os

import torch
from torch.utils.data import Dataset
# from torch.distributions.multivariate_normal import MultivariateNormal
# from torch.distributions.multinomial import Multinomial

from sklearn.datasets import make_blobs, make_multilabel_classification, make_gaussian_quantiles

import numpy as np
import re
from tqdm.contrib import tzip

class GaussianDataset(Dataset):
  def __init__(self, n_features, data_len):
    # self.shape = shape
    self.n_features = n_features
    self.data_len = data_len
    self.xs, self.labels = self._gaussian_data()
    assert len(self.xs) == self.data_len

  def __getitem__(self, index):
    x, label = self.xs[index], self.labels[index]
    return x, label

  def __len__(self):
    return len(self.xs)

  def _gaussian_data(self):
    # X, y = make_blobs(n_samples=self.data_len, centers=4, n_features=self.n_features, random_state=0)
    X, y = make_gaussian_quantiles(
      n_samples=self.data_len,
      n_features=self.n_features,
      n_classes = 4,
      random_state = 0,
    )
    return torch.from_numpy(X).float(), torch.from_numpy(y)

  # def _gaussian_data(self):
  #   mean_11 = torch.zeros(self.n)
  #   self.m_11 = MultivariateNormal(mean_11, torch.eye(self.n))
  #   mean_12 = 2 * torch.ones(self.n)
  #   self.m_12 = MultivariateNormal(mean_12, torch.eye(self.n))
  #   mean_01 = torch.zeros(self.n)
  #   mean_01[:self.n // 2] += 2 * torch.ones(self.n // 2)
  #   self.m_01 = MultivariateNormal(mean_01, torch.eye(self.n))
  #   mean_02 = 2 * torch.ones(self.n)
  #   mean_02[:self.n // 2] += 2 * torch.ones(self.n // 2)
  #   self.m_02 = MultivariateNormal(mean_02, torch.eye(self.n))
    
  #   m = Multinomial(self.data_len, torch.tensor([1., 1., 1., 1.]))
  #   cluster_lens = m.sample()

  #   xs = torch.empty(0, self.n)
  #   labels = torch.Tensor(0)
  #   clusters = torch.Tensor(0)
  #   for m, c, l, cluster_len in tzip(
  #     [self.m_11, self.m_12, self.m_01, self.m_02],
  #     [1, 2, 3, 4],
  #     [1, 1, 0, 0],
  #     list(cluster_lens)
  #   ):
  #     label_shape = [(int(cluster_len.item()))]
  #     xs_new = m.sample(sample_shape=label_shape)
  #     xs = torch.cat([xs, xs_new], dim = 0)
  #     labels = torch.cat([labels, torch.full(label_shape, l)], dim = 0)
  #     clusters = torch.cat([clusters, torch.full(label_shape, c)], dim = 0)

  #   return xs, labels.int(), clusters.int()