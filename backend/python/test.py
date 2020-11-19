import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from utils import convNet


# setting paths and some initial parameter
PATH = os.path.join(os.getcwd(), 'backend', 'python')
TRAIN_DIR = os.path.join(PATH, "dataset", "train")
TEST_DIR = os.path.join(PATH, "dataset", "test")
CLASSES = 'Cat', 'Dog'
BATCH_SIZE = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformations
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(
    (128, 128)), transforms.RandomHorizontalFlip(0.5)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((128, 128))])

# loading data
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

# hyperparamters
NUM_EPOCHS = 10
ITER_PER_EPOCH = math.ceil(len(train_set)/BATCH_SIZE)
LEARNING_RATE = 0.01
PATH_CHECKPOINT = os.path.join(PATH, "checkpoint_dict_model.pt")
PATH_MODEL = os.path.join(PATH, "state_dict_model.pt")

# initialize model, optimizer and loss criterion
model = convNet().to(device)
# load model for validation set
model.load_state_dict(torch.load(os.path.join(
    os.getcwd(), 'backend', 'python', 'state_dict_model.pt')))

optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# training loop
n_correct = 0
n_samples = 0
model.train()
for epoch in range(NUM_EPOCHS):
    for i, (features, label) in enumerate(train_loader):
        features = features.to(device)
        label = label.to(device)
        predictions = model(features)

        loss = criterion(predictions.to(torch.float32),
                         label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sm = nn.Softmax()
        prediction_cls = sm(predictions)
        _, prediction_cls = torch.max(prediction_cls, 1)
        n_correct += torch.sum(prediction_cls == label.data)
        n_samples += label.shape[0]
        if i % 20 == 0:
            print(
                f'Epoch: {epoch+1}/{NUM_EPOCHS}, Iteration: {i+1}/{ITER_PER_EPOCH} Accuracy: {n_correct/n_samples:.4%}')

# save model
torch.save(model.state_dict(),  PATH_MODEL)
# save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, PATH_CHECKPOINT)

# test loop
n_correct = 0
n_samples = 0
model.eval()
with torch.no_grad():
    for i, (features, label) in enumerate(test_loader):
        features = features.to(device)
        label = label.to(device)
        predictions = model(features)
        _, prediction_cls = torch.max(predictions, 1)
        n_correct += (prediction_cls == label).sum().item()
        n_samples += label.shape[0]
    print(f'Accuracy: {n_correct/n_samples:.4%}')
