import os
import random
import re
import time

import cv2
import numpy as np
import torch as torch
import torchvision
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

IMG_DIR_5_0 = 'duibu/product/images_5_0/'

BATCH_SIZE = 128
LEARNING_RATE = 0.1
EPOCHS = 30
GAMMA = 0.7


def log(s):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}")


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()
    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.9, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class APP_MATCHER(Dataset):
    def __init__(self, dir, train):
        super(APP_MATCHER, self).__init__()
        self.dir = dir
        allImageNames = os.listdir(dir)
        index = int(len(allImageNames) * 0.8)
        self.imageNames = allImageNames[:index] if train else allImageNames[index:]
        imgLen = len(self.imageNames)
        self.ids = []
        self.data = []
        for imageName in self.imageNames:
            matchObj = re.match(r'(\d+_.+)_(\d+)_(\d+)_(\d+).jpg', imageName)
            self.ids.append(matchObj.group(1))
            img = cv2.imread(IMG_DIR_5_0 + imageName, cv2.IMREAD_GRAYSCALE)
            self.data.append(np.array([cv2.resize(img, (224, 224))]))
        self.info = []
        for i in range(imgLen):
            count = 0
            while True:
                j = i + count
                if j < imgLen and self.ids[i] == self.ids[j]:
                    self.info.append((i, j, 1))
                    count += 1
                else:
                    while True:
                        if count == 0: break
                        k = random.randint(0, imgLen - 1)
                        if self.ids[i] != self.ids[k]:
                            self.info.append((i, k, 0))
                            count -= 1
                    break

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        img_1_index, img_2_index, target = self.info[index]
        image_1 = torch.tensor(self.data[img_1_index], dtype=torch.float)
        image_2 = torch.tensor(self.data[img_2_index], dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return image_1, image_2, target


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    log('start')
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    torch.manual_seed(1)
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f'device {device}')

    # imageNames = os.listdir(IMG_DIR_5_0)
    # X = []
    # Y = []
    # for index, imageName in enumerate(imageNames):
    #     matchObj = re.match(r'(\d+_.+)_(\d+)_(\d+)_(\d+).jpg', imageName)
    #     id = matchObj.group(1)
    #     path = IMG_DIR_5_0 + imageName
    #     X.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    #     Y.append(id)
    # classes = list(set(Y))
    # log(f'class num = {len(classes)}')
    # X = np.array(X)
    # Y = [classes.index(y) for y in Y]
    # Y = np.array(Y)
    # log(f'X shape = {X.shape}')
    # log(f'Y shape = {Y.shape}')
    # log(f'X = {X}')
    # log(f'Y = {Y}')

    train_dataset = APP_MATCHER(IMG_DIR_5_0, True)
    log(f'train_dataset product size  {len(set(train_dataset.ids))}')
    log(f'train_dataset img size  {len(train_dataset.imageNames)}')
    log(f'train_dataset data size {train_dataset.__len__()}')

    test_dataset = APP_MATCHER(IMG_DIR_5_0, False)
    log('')
    log(f'test_dataset product size  {len(set(test_dataset.ids))}')
    log(f'test_dataset img size  {len(test_dataset.imageNames)}')
    log(f'test_dataset data size {test_dataset.__len__()}')
    log('')
    train_loader = torch.utils.data.DataLoader(train_dataset, **{'batch_size': BATCH_SIZE})
    test_loader = torch.utils.data.DataLoader(test_dataset, **{'batch_size': BATCH_SIZE})

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    test(model, device, train_loader)
    test(model, device, test_loader)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "main2.pt")
