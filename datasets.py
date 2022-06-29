import os

import torch
import numpy as np
from torchvision import datasets
import re

from torchvision.datasets.utils import download_url

class LFWPeopleAttribute(datasets.LFWPeople):
  def __init__(
    self,
    root,
    split = '10fold',
    image_set = 'funneled',
    transform = None,
    target_transform = None,
    download = False,
    attribute_classes = None,
    target_class = None,
    inference_class = None,
  ):
    self.attributes_file = f"lfw_attributes.txt"
    super().__init__(root, split, image_set, transform, target_transform, download)
    # todo - check integrity
    # get attributes
    self.attribute_class_selects = attribute_classes
    self.target_class = target_class
    self.inference_class = inference_class
    self.attribute_class_to_idx = self._get_attribute_class()

    self.target_to_attribute, self.attribute_class_selects_idxs = self._get_target_to_attributes()
    # todo : change target to identity
    # better variable name for identtiy, target, attribute

    self.data, self.targets, self.attributes = self._normalize_data()
    self.identities = self.targets
    if self.target_class is not None:
      # self.attribute_target_index = self.attribute_class_selects_idxs[self.target_class]
      self.targets, self.attributes = self._get_attribute_target()

  def download(self):
    super().download()
    download_url(f"https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt", self.root)

  def _get_attribute_class(self):
    with open(os.path.join(self.root, self.attributes_file)) as f:
      for i, line in enumerate(f):
        if i == 1:
          attribute_classes = line.strip().split('\t')[3:]
          break
        continue
    return {attribute_class : i for i, attribute_class in enumerate(attribute_classes)}

  def _get_target_to_attributes(self):
    target_to_attribute = {}

    with open(os.path.join(self.root, self.attributes_file)) as f:
      lines = f.readlines()
      attribute_classes = lines[1].strip().split('\t')[3:]
      
      attribute_class_selects_idxs = {attr: self.attribute_class_to_idx[attr] for attr in self.attribute_class_selects}

      for line in lines[2:]:
        split_lines = line.strip().split("\t")
        identity = split_lines[0]
        identity = re.sub(" ", "_", identity)
        idx = self.class_to_idx[identity]
        identity_attribute = [float(split_lines[2:][i]) for i in attribute_class_selects_idxs.values()]
        target_to_attribute[idx] = identity_attribute

      return target_to_attribute, attribute_class_selects_idxs

  def _get_binary_attribute_target(self):
    attributes = np.array(self.attributes)
    if self.target_class == 'gender':
      attribute_idx = self.attribute_class_selects_idxs['Male']
    elif self.target_class == 'smile':
      attribute_idx = self.attribute_class_selects_idxs['Smiling']
    attributes = attributes[:, attribute_idx]
    targets = np.sign(attributes)
    targets = targets.astype(np.int32)
    targets[targets == -1] = 0

    return targets

  def _get_multiple_attribute_target(self):
    attributes = np.array(self.attributes)
    if self.target_class == 'race':
      attribute_idxs = [self.attribute_class_selects_idxs[attr] for attr in ['Asian', 'White', 'Black']]
    elif self.target_class == 'age':
      attribute_idxs = [self.attribute_class_selects_idxs[attr] for attr in ['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior']]
    elif self.target_class == 'hair':
      attribute_idxs = [self.attribute_class_selects_idxs[attr] for attr in ['Black Hair', 'Blond Hair', 'Brown Hair', 'Bald']]
    elif self.target_class == 'eyewear':
      attribute_idxs = [self.attribute_class_selects_idxs[attr] for attr in ['No Eyewear', 'Eyeglasses', 'Sunglasses']]
    attributes[:, attribute_idxs]
    targets = np.argmax(attributes, axis=-1)

    return targets

  def _get_attribute_target(self):
    if self.target_class in ['gender', 'smile']:
      targets = self._get_binary_attribute_target()
    elif self.target_class in ['race', 'age', 'hair', 'eyewear']:
      targets = self._get_multiple_attribute_target()

    attributes = np.array(self.attributes)
    if self.inference_class is not None:
      attributes = attributes[:, self.attribute_class_selects_idxs[self.inference_class]]

    return targets, attributes


  def _normalize_data(self):
    normalized_datas = []
    normalized_targets = []
    normalized_attributes = []
    for data, target in zip(self.data, self.targets):
      if target in self.target_to_attribute:
        normalized_datas.append(data)
        normalized_targets.append(target)
        normalized_attributes.append(self.target_to_attribute[target])
    return normalized_datas, normalized_targets, normalized_attributes

  def __getitem__(self, index):
    img = self._loader(self.data[index])
    attribute = self.attributes[index]
    target = self.targets[index]

    if self.transform is not None:
       img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target, attribute
