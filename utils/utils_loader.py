from torchvision import transforms
import numpy as np
import os
from PIL import Image

data_transform_rn152 = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

def img_name_to_PIL_img(img_name, img_dir, transform=None):
  img_name = img_name.replace('img', 'img_reduced') + '.npy'
  img_path = os.path.join(img_dir, img_name)
  arr = np.load(img_path)
  img = Image.fromarray(arr)
  if transform:
    img = transform(img)
  return img
