import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from convNetModel import convNet


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
    plt.title(label[i])
# plt.show()

# hyperparamters
NUM_EPOCHS = 2
ITER_PER_EPOCH = math.ceil(len(train_set)/BATCH_SIZE)
LEARNING_RATE = 0.01

model = convNet().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

n_correct = 0
n_samples = 0
model.train()
for epoch in range(NUM_EPOCHS):
    for i, (features, label) in enumerate(train_loader):
        features, label = features.to(
            device), label.to(torch.float32).to(device)
        predictions = model(features)
        predictions = predictions.view(-1).to(torch.float32)
        loss = criterion(predictions, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction_cls = predictions < 0.5
        n_correct += (prediction_cls == label).sum().item()
        n_samples += label.shape[0]
        if i % 300 == 0:
            print(
                f'Epoch: {epoch+1}/{NUM_EPOCHS}, Iteration: {i+1}/{ITER_PER_EPOCH} Accuracy: {n_correct/n_samples:.4%}')

n_correct = 0
n_samples = 0
model.eval()
with torch.no_grad():
    for i, (features, label) in enumerate(test_loader):
        features = features.to(device)
        label = label.to(device)
        predictions = model(features)
        predictions = predictions.view(-1)
        prediction_cls = predictions < 0.5
        n_correct += (prediction_cls == label).sum().item()
        n_samples += label.shape[0]
    print(f'Accuracy: {n_correct/n_samples:.4%}')
