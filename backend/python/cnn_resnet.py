import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
import math

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

# hyperparamters
NUM_EPOCHS = 2
ITER_PER_EPOCH = math.ceil(len(train_set)/BATCH_SIZE)
LEARNING_RATE = 0.01
PATH_CHECKPOINT = "checkpoint_resnet_model.pt"
PATH_MODEL = "state_resnet_model.pt"

# load resnet model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

optimizer = torch.optim.SGD(
    params=model.parameters(), lr=LEARNING_RATE, momemtum=0.9)
criterion = nn.CrossEntropyLoss()
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

n_correct = 0
n_samples = 0
model.train()
for epoch in range(NUM_EPOCHS):
    for i, (features, label) in enumerate(train_loader):
        features, label = features.to(
            device), label.to(device)
        predictions = model(features)

        loss = criterion(predictions.to(torch.float32),
                         label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()

        _, prediction_cls = torch.max(predictions, 1)
        n_correct += torch.sum(prediction_cls == label.data)
        n_samples += label.shape[0]
        if i % 300 == 0:
            print(
                f'Epoch: {epoch+1}/{NUM_EPOCHS}, Iteration: {i+1}/{ITER_PER_EPOCH} Accuracy: {n_correct/n_samples:.4%}')


# save model
torch.save(model.state_dict(), PATH_MODEL)
# save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, PATH_CHECKPOINT)


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
