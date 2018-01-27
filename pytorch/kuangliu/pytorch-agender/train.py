from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from net import AGNet
from loss import AGLoss
from datagen import ListDataset

from torch.autograd import Variable

import missinglink

OWNER_ID = '764e5d22-7128-a8cf-4213-c49beec73998'
PROJECT_TOKEN = 'UJrhwlpluaSbrWew'
HOST = 'https://missinglink-staging.appspot.com'

missinglink_project = missinglink.PyTorchProject(owner_id=OWNER_ID, project_token=PROJECT_TOKEN, host=HOST)


parser = argparse.ArgumentParser(description='PyTorch AGNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_correct = 0 # best number of age_correct + gender_correct
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
list_files_root = '../../../AdienceFaces/folds/train_val_txt_files_per_fold/test_fold_is_0'
images_root = '../../../AdienceFaces/DATA/aligned'

transform_train = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.RandomCrop(150, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
trainset = ListDataset(root=images_root, list_file=os.path.join(list_files_root, 'age_gender_train_subset.txt'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

transform_test = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
testset = ListDataset(root=images_root, list_file=os.path.join(list_files_root, 'age_gender_test.txt'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

# Model
net = AGNet()
# net.load_state_dict(torch.load('./model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_correct = checkpoint['correct']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net)  # , device_ids=range(torch.cuda.device_count()))
# net.cuda()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    age_correct = 0
    gender_correct = 0
    for batch_idx, (inputs, age_targets, gender_targets) in experiment.batch_loop(iterable=trainloader):
        inputs = Variable(inputs)  # .cuda())
        age_targets = Variable(age_targets)  # .cuda())
        gender_targets = Variable(gender_targets).float()  # .cuda()).float()
        optimizer.zero_grad()
        age_preds, gender_preds = net(inputs)
        loss = wrapped_criterion(age_preds, age_targets, gender_preds, gender_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        age_correct_i, gender_correct_i = accuracy(
            age_preds, age_targets, gender_preds, gender_targets)
        age_correct += age_correct_i
        gender_correct += gender_correct_i
        total += len(inputs)
        print('train_loss: %.3f | avg_loss: %.3f | age_prec: %.3f (%d/%d) | gender_prec: %.3f (%d/%d)  [%d/%d]' \
              % (loss.data[0], wrapped_average_loss(train_loss, batch_idx + 1), \
                 wrapped_age_accuracy(age_correct, total), age_correct, total, \
                 wrapped_gender_accuracy(gender_correct, total), gender_correct, total, \
                 batch_idx + 1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    total = 0
    age_correct = 0
    gender_correct = 0
    with experiment.test(test_iterations=len(testloader)):
        for batch_idx, (inputs, age_targets, gender_targets) in enumerate(testloader):
            inputs = Variable(inputs, volatile=True)  # .cuda(), volatile=True)
            age_targets = Variable(age_targets)  # .cuda())
            gender_targets = Variable(gender_targets).float()  # .cuda()).float()

            age_preds, gender_preds = net(inputs)

            experiment.confusion_matrix(target=age_targets, output=age_preds)

            loss = wrapped_criterion(age_preds, age_targets, gender_preds, gender_targets)

            test_loss += loss.data[0]
            age_correct_i, gender_correct_i = accuracy(
                age_preds, age_targets, gender_preds, gender_targets)
            age_correct += age_correct_i
            gender_correct += gender_correct_i
            total += len(inputs)
            print('test_loss: %.3f | avg_loss: %.3f | age_prec: %.3f (%d/%d) | gender_prec: %.3f (%d/%d)  [%d/%d]' \
                % (loss.data[0], wrapped_average_loss(test_loss, batch_idx+1),      \
                   wrapped_age_accuracy(age_correct, total), age_correct, total,  \
                   wrapped_gender_accuracy(gender_correct, total), gender_correct, total, \
                   batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_correct
    if age_correct + gender_correct > best_correct:
        print('Saving..')
        best_correct = age_correct + gender_correct
        state = {
            'net': net.module.state_dict(),
            'correct': best_correct,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

def accuracy(age_preds, age_targets, gender_preds, gender_targets):
    '''Measure batch accuracy.'''
    AGE_TOLERANCE = 5
    age_prob = F.softmax(age_preds)
    variable = Variable(torch.arange(1, 101))  # .cuda()
    age_expect = torch.sum(variable*age_prob, 1)
    age_correct = ((age_expect-age_targets.float()).abs() < AGE_TOLERANCE).long().sum().cpu().data[0]

    gender_preds = F.sigmoid(gender_preds)
    gender_preds = (gender_preds > 0.5).long()
    gender_correct = (gender_preds == gender_targets.long()).long().cpu().sum().data[0]
    return age_correct, gender_correct


def accuracy_precent(correct, total):
    return 100. * correct / total


def average(total, count):
    return total / count


with missinglink_project.create_experiment(
    net,
    display_name='Age Gender PyTorch',
    optimizer=optimizer,
    metrics={
        'Age Accuracy': accuracy_precent,
        'Gender Accuracy': accuracy_precent,
        'Average Loss': average
    },
) as experiment:
    wrapped_criterion = experiment.wrap_metrics({'Total Loss': AGLoss(experiment)})['Total Loss']
    wrapped_age_accuracy = experiment.metrics['Age Accuracy']
    wrapped_gender_accuracy = experiment.metrics['Gender Accuracy']
    wrapped_average_loss = experiment.metrics['Average Loss']

    for _, epoch in experiment.epoch_loop(iterable=range(start_epoch, start_epoch+200)):
        train(epoch)
        test(epoch)
