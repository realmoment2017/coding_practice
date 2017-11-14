import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=(2,2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1,1))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=(1,1))
        self.bn4 = nn.BatchNorm2d(512)
        self.fc5 = nn.Linear(131072, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(2048, 1)

    def net(self,x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.bn5(F.relu(self.fc5(x)))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, y):
        output1 = self.net(x)
        output2 = self.net(y)
        output_c = torch.cat((output1, output2), 1)
        output = self.fc(output_c)
        output = torch.sigmoid(output)
        return output


class Dataset_lfw(Dataset):
    def __init__(self, root_dir, txt_file, is_transform=True):
        """
        Args:
            txt_file (string): Path to the txt file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img1_list, self.img2_list, self.label_list = self.__readfile__(txt_file)
        self.root_dir = root_dir
        self.is_transform = is_transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.img1_list[idx])
        img1 = plt.imread(img1_path)
        img2_path = os.path.join(self.root_dir, self.img2_list[idx])
        img2 = plt.imread(img2_path)
        labels = self.label_list[idx]
        img1 = np.asarray(img1,dtype=np.float32)
        img2 = np.asarray(img2,dtype=np.float32)
        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))
        sample = {'image1': img1, 'image2': img2, 'labels': labels}
        if self.is_transform:
            sample = self.pre_transform(sample)
        return sample

    def pre_transform(self,sample):
        sample['image1'] = np.resize(sample['image1'], (3, 128, 128))
        sample['image2'] = np.resize(sample['image2'], (3, 128, 128))
        if sample['labels'] == '0':
            sample['labels'] = torch.from_numpy(np.asarray([0],dtype=np.float32))
        else:
            sample['labels'] = torch.from_numpy(np.asarray([1],dtype=np.float32))
        return sample

    def __readfile__(self, txt_file):
        img1_dir = []
        img2_dir = []
        label = []
        with open(txt_file, 'r') as f:
            # print(f)
            # i = 0
            for line in f:
                # i = i+1
                # if i > 20:
                #     break
                data = line.strip()
                data = data.split(' ')
                img1_dir.append(data[0])
                img2_dir.append(data[1])
                label.append(data[2])
        return img1_dir, img2_dir, label

    def show_pair(self, idx):
        print('length of the dataset: ', len(self))
        sample = self[idx]
        fig = plt.figure()
        sample['image1'] = np.transpose(sample['image1'], (1, 2, 0))
        sample['image2'] = np.transpose(sample['image2'], (1, 2, 0))
        # print(sample['image1'].shape, sample['image2'].shape, sample['labels'])
        fig = plt.subplot(1, 2, 1)
        plt.imshow(sample['image1'])
        fig = plt.subplot(1, 2, 2)
        plt.imshow(sample['image2'])
        plt.show()
        plt.tight_layout()
        fig.set_title('Sample #{}'.format(idx))
        fig.axis('off')

train_data_root_dir = './lfw'
train_data_txt_dir = './lfw/train.txt'
test_data_root_dir = './lfw'
test_data_txt_dir = './lfw/test.txt'

# model.cuda()
# print(model)
# params = list(model.parameters())
# print(len(params))

trainset = Dataset_lfw(train_data_root_dir, train_data_txt_dir)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = Dataset_lfw(test_data_root_dir, test_data_txt_dir)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# data_transform = transforms.Compose([
#         transforms.RandomSizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
# trainset.show_pair(1654)


# for i, data in enumerate(trainloader, 0):
#     model = Net()
#     criterion = nn.BCELoss()
#
#     labels = data['labels']
#     img1 = data['image1']
#     img2 = data['image2']
#     print(img2.size())
#     inputs1, inputs2, labels = Variable(img1), Variable(img2), Variable(labels)
#     # outputs = model(inputs1, inputs2)
#     outputs = model(inputs1, inputs2)
#     print(outputs)
#     print(labels)
#     loss = criterion(outputs, labels)
#     if i==2:
#         break


def train(trainloader):
    plot_x = []
    plot_y = []
    model = Net().cuda()
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    print(time.clock())
    print('Start Training')
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            labels = data['labels']
            img1 = data['image1']
            img2 = data['image2']
            # wrap them in Variable
            inputs1, inputs2, labels = Variable(img1.cuda()), Variable(img2.cuda()), Variable(labels.cuda())
            # inputs1, inputs2, labels = Variable(img1), Variable(img2), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            plot_x.append((epoch + 1) * (i + 1))
            plot_y.append(running_loss)
            if i % 100 == 99:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        plt.plot(plot_x, plot_y, 'bo')
        plt.show()
    print('Finished Training')
    print(time.clock())
    torch.save(model.state_dict(), './p1a.pkl')




def test(testloader):
    model = Net().cuda()
    correct = 0
    total = 0
    print(time.clock())
    model.load_state_dict(torch.load('./p1a.pkl'))
    print('Start Testing')
    for i, data in enumerate(testloader, 0):
        print(i)
        labels = data['labels']
        # print(labels)
        img1 = data['image1']
        img2 = data['image2']
        # wrap them in Variable
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs1, inputs2 = Variable(img1), Variable(img2)
        outputs = model(inputs1, inputs2)
        # print(outputs)
        predicted = outputs.data>0.5
        predicted = predicted.type('torch.LongTensor')
        labels = labels.type('torch.LongTensor')
        correct += torch.sum(predicted == labels)
        total += labels.size(0)
    print('Accuracy of the network on the all test images: %d %%' % (100 * correct / total))
    print('Finished Testing')




train(trainloader)
# print(len(testloader))
# test(testloader)
