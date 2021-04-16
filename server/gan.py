import time
import argparse
import matplotlib
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from models import *
import matplotlib.pyplot as plt

def train_c(train_loader, model, device, criterion, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 200 == 0 and i != 0:
            print('Train Epoch # {} [{:>5}/{}]\tloss: {:.6f}'.format(epoch, i * len(data), len(train_loader.dataset),
                                                                     train_loss / 200))
            scheduler.step(train_loss / 200)
            train_loss = 0


def test_c(test_loader, model, device, epoch):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        correct_2 = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_2 += (predicted == 10).sum().item()

        print('\t==============================================================> Test Epoch #{:>2}: {}({})/{} ({:>3.2f}%)'.format(epoch, correct, correct + correct_2, total, 100. * correct / total))


if __name__ == '__main__':

    batch_size = 64
    nz = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    train_data = torchvision.datasets.MNIST(
        train=True,
        download=True,
        root='./data',
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.MNIST(
        train=False,
        download=True,
        root='./data',
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################################
    classifier = MnistNetTiny(num_classes=11)
    classifier.load_state_dict(torch.load('logs/mnist_tiny_11.pth'))
    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)
    classifier.to(device)

    optimizer_c = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9)
    scheduler_c = optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, 'min', factor=0.2, verbose=1)
    criterion_c = nn.CrossEntropyLoss()
    # 重写一种损失函数，使得分类不准确的结果分到10

    ###############################################################################################
    generator = Generator(num_classes=10, nz=nz)
    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
    generator.to(device)

    optimizer_g = optim.SGD(generator.parameters(), lr=0.00002, momentum=0.9)
    criterion_g = nn.CrossEntropyLoss()

    g_labels = torch.tensor([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                0, 1, 2, 3]).to(device)

    fake_labels = torch.zeros(batch_size).to(device)
    fake_labels = fake_labels.fill_(10).long()
    # TODO
    # 1. 仅训练生成器，然后可视化CNN网络，看生成器生成的图片在网络中被提取了什么特征
    # 2. 再训练CNN，查看这和上面的对比有什么区别
    plt.ion()
    for epoch in range(3):
        for i, (images, labels) in enumerate(train_loader):

            #########################################################
            # real data
            # images, labels = images.to(device), labels.to(device)

            # z = torch.mul(torch.randn(batch_size, nz), 0.1).to(device)
            z = torch.randn(batch_size, nz).to(device)
            fake_images = generator(z, g_labels)
               
            # classifier.zero_grad()

            # if i % 10 == 0:
            #     # fake data
            #     output = classifier(fake_images.detach())
            #     loss_c_fake = criterion_c(output, fake_labels)
            #     loss_c_fake.backward()

                # output = classifier(fake_images.detach())
                # loss_c_fake_ = criterion_c(output, g_labels)
                # loss = loss_c_fake + loss_c_fake_
                # loss.backward()

            
            # output = classifier(images)
            # # print(torch.max(output, 1).indices)
            # loss_c_real = criterion_c(output, labels)
            # loss_c_real.backward()
            # optimizer_c.step()

            #########################################################

            if (i + 1) % 1 == 0:
                optimizer_g.zero_grad()
                output = classifier(fake_images)
                maxs = torch.max(output, 1)
                # print(maxs.indices[0:10])
                loss_g = criterion_g(output, g_labels)

                loss_g.backward()
                optimizer_g.step()
                print(f'epoch #{epoch:>2d}:{loss_g.item()} \t{maxs.indices[0:10]}')
                image = fake_images.detach().cpu().numpy()[0:10].reshape(28 * 10, 28)
                plt.imshow(image, cmap=plt.get_cmap('RdBu'))
                plt.pause(0.0001)
                plt.cla()
                torchvision.utils.save_image(fake_images.detach(), 'logs/fake_gan.png', normalize=True)

            # if (i + 1) % 99 == 0:
                # test_c(test_loader, classifier, device, 0)

    torch.save(classifier.state_dict(), f'logs/mnist_classifier_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.pth')
    torch.save(generator.state_dict(), f'logs/mnist_generator_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.pth')