import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
from convNetModel import convNet

model = convNet()
print(model)
TRAIN_DIR = "./dataset/train"
TEST_DIR = "./dataset/test"
CLASSES = 'Cat', 'Dog'
BATCH_SIZE = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(
    (128, 128)), transforms.RandomHorizontalFlip(0.5)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((128, 128))])

train_set = ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_set = ImageFolder(root=TEST_DIR, transform=test_transform)

train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

train_iter = iter(train_loader)
inputs, label = train_iter.next()
for i in range(BATCH_SIZE):
    plt.subplot(2, 3, i+1)
    plt.imshow(inputs[i][0])
    plt.title(CLASSES[label[i]])
# plt.show()
