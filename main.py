import json
import os
import re
import time

import cv2
import numpy as np
import requests
import torch as torch
import torchvision
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

Authorization = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2ODYxMDEwMDcsInVzZXJuYW1lIjoibGlueGluZyJ9.O1yDvcXN-LqiCa8qfomLYiIw8PAPg79Njj6c8e2j3Fg'
SIZE = 30
IMG_HOST = 'https://img.duibu.cn/'
IMG_DIR = 'duibu/product/images/'
IMG_DIR_5_0 = 'duibu/product/images_5_0/'

BATCH_SIZE = 64
LEARNING_RATE = 1.0
EPOCHS = 1
GAMMA = 0.7


def log(s):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}")


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
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


class APP_MATCHER(Dataset):
    def __init__(self, dir, train):
        super(APP_MATCHER, self).__init__()
        self.dir = dir
        allImageNames = os.listdir(dir)
        index = int(len(allImageNames) * 0.8)
        self.imageNames = allImageNames[:index] if train else allImageNames[index:]
        self.data = []
        self.ids = []
        for index, imageName in enumerate(self.imageNames):
            matchObj = re.match(r'(\d+_\d+)_(\d+)_(\d+)_(\d+).jpg', imageName)
            if matchObj:
                img = cv2.imread(IMG_DIR_5_0 + imageName, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                imgData = [img]
                self.ids.append(matchObj.group(1))
                self.data.append(imgData)
        self.data = np.array(self.data)
        self.ids = np.array(self.ids)
        self.info = []
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                target = 1 if self.ids[i] == self.ids[j] else 0
                self.info.append((i, j, target))

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        img_1_index, img_2_index, target = self.info[index]
        image_1 = self.data[img_1_index]
        image_2 = self.data[img_2_index]
        image_1 = torch.tensor(image_1, dtype=torch.float)
        image_2 = torch.tensor(image_2, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return image_1, image_2, target


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
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def download():
    url = 'https://new.duibu.cn/manage-api/product/list'
    header = {
        'Authorization': Authorization
    }
    body = {
        "deviceDescVersion": "3.0.0",
        "page": 1,
        "size": SIZE
    }
    response = requests.post(url, headers=header, json=body)
    products = json.loads(response.text)['data']['content']
    with open('products.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(products, ensure_ascii=False))
    log(f'products size = {len(products)}')
    imagePaths = []
    for product in products:
        for productImages in product['productImages']:
            imagePaths.append(productImages['storagePath'])
    log(f'imagePaths size = {len(imagePaths)}')
    with open('imagePaths.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(imagePaths, ensure_ascii=False))
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    for index, path in enumerate(imagePaths):
        path = path[1:]
        if os.path.exists(path):
            log(f'{path} already exists')
            continue
        url = IMG_HOST + path
        log(f'download index={index} {url}')
        r = requests.get(url)
        with open(path, 'wb') as file:
            file.write(r.content)
        # time.sleep(1)


def pick():
    if not os.path.exists(IMG_DIR_5_0):
        os.mkdir(IMG_DIR_5_0)
    imageNames = os.listdir(IMG_DIR)
    labels = set()
    for index, imageName in enumerate(imageNames):
        matchObj = re.match(r'(\d+_\d+)_(\d+)_(\d+)_(\d+).jpg', imageName)
        if matchObj.group(2) == '5' and matchObj.group(3) == '0':
            labels.add(matchObj.group(1))
            oldPath = IMG_DIR + imageName
            newPath = IMG_DIR_5_0 + imageName
            if os.path.exists(newPath):
                log(f'{newPath} already exists')
                continue
            log(f'index={index} saved {newPath}')
            imread = cv2.imread(oldPath)
            cv2.imwrite(newPath, imread)


if __name__ == '__main__':
    log('start')
    # download()
    # pick()

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    torch.manual_seed(1)
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_dataset = APP_MATCHER(IMG_DIR_5_0, True)
    test_dataset = APP_MATCHER(IMG_DIR_5_0, False)
    train_loader = torch.utils.data.DataLoader(train_dataset, **{'batch_size': BATCH_SIZE})
    test_loader = torch.utils.data.DataLoader(test_dataset, **{'batch_size': BATCH_SIZE})

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    torch.save(model.state_dict(), "siamese_network.pt")
