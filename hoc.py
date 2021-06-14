import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn import metrics


def get_confusion_matrix(y_true, y_pred, normalize=False):
    y_true_cm = []
    y_pred_cm = []
    for e in y_true:
        if e == 0:
            y_true_cm.append([1, 0, 0, 0, 0, 0])
        elif e == 1:
            y_true_cm.append([0, 1, 0, 0, 0, 0])
        elif e == 2:
            y_true_cm.append([0, 0, 1, 0, 0, 0])
        elif e == 3:
            y_true_cm.append([0, 0, 0, 1, 0, 0])
        elif e == 4:
            y_true_cm.append([0, 0, 0, 0, 1, 0])
        elif e == 5:
            y_true_cm.append([0, 0, 0, 0, 0, 1])
    y_true_cm = np.array(y_true_cm)
    for e in y_pred:
        if e == 0:
            y_pred_cm.append([1, 0, 0, 0, 0, 0])
        elif e == 1:
            y_pred_cm.append([0, 1, 0, 0, 0, 0])
        elif e == 2:
            y_pred_cm.append([0, 0, 1, 0, 0, 0])
        elif e == 3:
            y_pred_cm.append([0, 0, 0, 1, 0, 0])
        elif e == 4:
            y_pred_cm.append([0, 0, 0, 0, 1, 0])
        elif e == 5:
            y_pred_cm.append([0, 0, 0, 0, 0, 1])
    y_pred_cm = np.array(y_pred_cm)

    cm = metrics.confusion_matrix(y_true_cm.argmax(axis=1), y_pred_cm.argmax(axis=1))
    print(cm)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    return cm


# labelpath = 'D:/PYc/Pytorch/graph_labels.txt'
labelpath = 'D:/PYc/Re/test_text.txt'
# labelpath = '/home/wh/Project/graphic-representations-classification/graph_labels.txt'
labels = []

with open(labelpath, 'r') as file:
    for line in file:
        labels.append(int(line.split()[0]))
print(set(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
EPOCH = 1
a = 0.010
# imagePath = 'D:/PYc/Test-images/'
imagePath = 'D:/PYc/Test-images2/'
# imagePath = '/home/wh/Project/graphic-representations-classification/Images/'
images = []

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = resnet18(pretrained=False, num_classes=6).to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

AllLabels = labels
lossList = []

for epoch in range(EPOCH):
    sum_loss = 0
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0

    for i in range(10):
        images = []
        for index in tqdm(range(i * 100, i * 100 + 100)):
            fileName = str(index) + '_' + str(AllLabels[index]) + '.jpg'
            image = Image.open(imagePath + fileName)
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            images.append(image)
        label = torch.from_numpy(np.array(AllLabels[i * 100: i * 100 + 100])).long()
        trainimg = images[:int(len(images)*0.8)]
        testimg = images[int(len(images)*0.8):]
        trainlabel = label[:int(len(label)*0.8)]
        testlabel = label[int(len(label)*0.8):]
        trainset = TensorDataset(torch.cat(trainimg, 0), trainlabel)
        testset = TensorDataset(torch.cat(testimg, 0), testlabel)

        trainloader = DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=0)
        testloader = DataLoader(dataset=testset, batch_size=100, shuffle=True, num_workers=0)

        # train
        for step, (inputs, label) in tqdm(enumerate(trainloader)):
            length = len(trainloader)
            inputs, label = inputs.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            print(loss.item())
            total_train += label.size(0)
            sum_loss += loss.item()
            lossList.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct_train += predicted.eq(label.data).cpu().sum()
        print('[epoch: %d ] Loss: %.03f | Acc: %.3f%%' % (epoch + 1, sum_loss / 50000, 100.0 * correct_train / total_train))

        # test
        for step, (inputs, label) in tqdm(enumerate(testloader)):
            length = len(trainloader)
            inputs, label = inputs.to(device), label.to(device)
            model.eval()
            outputs = model(inputs)
            total_test += label.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_test += predicted.eq(label.data).cpu().sum()
        print('[epoch: %d ] Acc: %.3f%%' % (epoch + 1, 100.0 * correct_test / total_test))

        confusion_matrix = get_confusion_matrix(label.tolist(), predicted.tolist(), normalize=True)
        print(confusion_matrix)
