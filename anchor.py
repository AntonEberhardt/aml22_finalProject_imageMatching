import os
import numpy as np
from skimage import io, transform
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import random
import json
import math
import matplotlib.pyplot as plt
import torchvision.models as models
import time
import pickle
import pdb

# from create_cambridge_scene.py
# Train and Test file paths for a particular Scene

class Resize(object):
    """ Resize the image"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, dof, orig = (
            sample["image"],
            sample["coordinates"],
            sample["dof"],
            sample["origxy"],
        )
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        img = transform.resize(image, (new_h, new_w), mode="constant")

        return {"image": img, "coordinates": label, "dof": dof, "origxy": orig}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, dof, orig = (
            sample["image"],
            sample["coordinates"],
            sample["dof"],
            sample["origxy"],
        )
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image, (2, 0, 1))
        # image = image[np.newaxis,:,:]
        return {
            "image": torch.from_numpy(image).float(),
            "coordinates": torch.from_numpy(np.array([label])).float(),
            "dof": torch.from_numpy(np.array([dof])).float(),
            "origxy": torch.from_numpy(np.array([orig])).float(),
        }


class Cambridge(torch.utils.data.Dataset):
    def __init__(self, gt_file, dataset_root, anchors, shuffle=False, transform=None):

        self.dataset_root = dataset_root
        # self.all_file_names = [dataset_root + x for x in os.listdir(dataset_root)]

        with open(gt_file) as f:
            temp_gt = json.load(f)

        # with open(anchors) as g:
        # 	temp_anchors = json.load(g)

        self.all_file_names = temp_gt[0]
        self.labels = temp_gt[1]
        print(np.array(self.labels, dtype = object).shape)
        self.dof = temp_gt[2]
        self.orig = temp_gt[3]
        # self.anch = temp_anchors[1]

        # self.labels = [sum(temp_gt[1][i],[]) for i in range(len(temp_gt[1]))]

        self.shuffle = shuffle

        if self.shuffle:
            temp = list(zip(self.all_file_names, self.labels, self.dof, self.orig))
            random.shuffle(temp)
            self.all_file_names, self.labels, self.dof, self.orig = zip(*temp)

        self.transform = transform

    def __len__(self):
        return len(self.all_file_names)

    def __getitem__(self, idx):

        # img_name = os.path.join(self.dataset_root, self.all_file_names[idx])
        image = io.imread(self.all_file_names[idx])
        sample = {
            "image": image,
            "coordinates": self.labels[idx],
            "dof": self.dof[idx],
            "origxy": self.orig[idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):
    def __init__(self, nAnchorpoints):
        super(Net, self).__init__()
        self.feat_extractor = model.features
        # self.feat_extractor = model.Conv2d_1a_3x3
        self.classifier = nn.Linear(2208, nAnchorpoints)
        self.regressor = nn.Linear(2208, 2*nAnchorpoints)
        self.dof_regressor = nn.Linear(2208, 4)

    def forward(self, x):
        out = self.feat_extractor(x)
        out = GAP(out)
        out = out.view(out.size(0), -1)
        classify = self.classifier(F.relu(out))
        regress = self.regressor(F.relu(out))
        dof_regress = self.dof_regressor(F.relu(out))
        return classify, regress, dof_regress


def custom_loss(classify, regress, labels, class_labels, dof, dof_regress, ceFactor=0, criterion=nn.CrossEntropyLoss(), nAnchorpoints = 33):
    dist = regress - labels
    dist = dist ** 2

    loss = 0.0

    for i in range(dist.size()[0]):
        for j in range(0, nAnchorpoints, 2):
            loss += (dist[i][j] + dist[i][j + 1]) * classify[i][int(j / 2)]

    _, class_labels = torch.min(class_labels, 1)
    _, pred_labels = torch.min(classify, 1)

    loss = (
        1. * loss + ceFactor * criterion(classify, class_labels) + 1. * mse(dof_regress, dof)
    )

    return loss

if __name__ == "__main__":
    train_gt_files = os.path.join("ShopFacade","dataset_train.txt") #Specify path to files for training
    test_gt_files = os.path.join("ShopFacade","dataset_test.txt")  #Specify path to files for testing

    f = open(train_gt_files,'r')
    train_gt_files = f.readlines()

    f = open(test_gt_files,'r')
    test_gt_files = f.readlines()

    train_gt_files = train_gt_files[3:]
    test_gt_files = test_gt_files[3:]

    dataset = [[],[],[],[]]

    for i in range(len(train_gt_files)):
        temp = train_gt_files[i].split(' ')
        temp = [j.strip() for j in temp]
        dataset[0].append(temp[0])
        temp2 = temp[4:]
        temp2 = [float(j) for j in temp2]
        temp = temp[1:3]
        temp = [float(j) for j in temp]
        dataset[1].append(temp)
        dataset[2].append(temp2)
        dataset[3].append(temp)

    # Dump dataset for training (specify path)

    with open(os.path.join("ShopFacade","dataset_train_mod.txt"),'w') as f:
        json.dump(dataset,f)


    dataset = [[],[],[],[]]

    for i in range(len(test_gt_files)):
        temp = test_gt_files[i].split(' ')
        temp = [j.strip() for j in temp]
        dataset[0].append(temp[0])
        temp2 = temp[4:]
        temp2 = [float(j) for j in temp2]
        temp = temp[1:3]
        temp = [float(j) for j in temp]
        dataset[1].append(temp)
        dataset[2].append(temp2)
        dataset[3].append(temp)

    # Dump dataset for testing (specify path)

    with open(os.path.join("ShopFacade","dataset_test_mod.txt"),'w') as f:
        json.dump(dataset,f)

    # from preprocess_cambridge_scene.py

    train_gt_files = "ShopFacade" #Path to ground truth for particular scene
    test_gt_files = "ShopFacade" #Path to ground truth for a particular scene

    f = open(os.path.join(train_gt_files, 'dataset_train_mod.txt'),'r')
    train = json.load(f)

    f = open(os.path.join(test_gt_files, 'dataset_test_mod.txt'),'r')
    test = json.load(f)

    train[0] = [os.path.join(train_gt_files, train[0][i]) for i in range(len(train[0]))]
    test[0] = [os.path.join(test_gt_files, test[0][i]) for i in range(len(test[0]))]

    ids = range(0,len(train[0]),10)

    #Precompute anchors

    anchors = [[],[]]

    for i in ids:
        anchors[0].append(train[0][i])
        anchors[1].append(train[1][i])

    # Save anchors

    nAnchors = len(anchors[0])

    with open(os.path.join(train_gt_files, 'anchors.txt'),'w') as f:
        json.dump(anchors, f)

    # Create and dump train, test data in the format for model input

    for i in range(len(train[0])):
        train[1][i] = (np.array(train[1][i]) - np.array(anchors[1]))
        train[1][i] = list(train[1][i])
        train[1][i] = [list(j) for j in train[1][i]]
        train[1][i] = sum(train[1][i],[])

    with open(os.path.join(train_gt_files, 'traindata.txt'),'w') as f:
        print(f"dumping {np.array(train, dtype=object).shape} shaped array")
        json.dump(train, f)

    # train[1] = sum(train[1],[])

    for i in range(len(test[0])):
        test[1][i] = (np.array(test[1][i]) - np.array(anchors[1]))
        test[1][i] = list(test[1][i])
        test[1][i] = [list(j) for j in test[1][i]]
        test[1][i] = sum(test[1][i],[])

    with open(os.path.join(test_gt_files, 'testdata.txt'),'w') as f:
        #print(f"dumping {np.array(test).shape} shaped array")
        json.dump(test, f)

    # test[1] = sum(test[1],[])

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    #==========DEVICE==============
    device = "cpu"

    batch_size = 16

    # composed = transforms.Compose([Rescale(256), RandomCrop(224),ToTensor()])
    composed = transforms.Compose([Resize(224), ToTensor()])

    train_dataset = Cambridge(
        os.path.join("ShopFacade","traindata.txt"),
        "ShopFacade",
        anchors=None, #"ShopFacade/anchors.txt",
        shuffle=True,
        transform=composed,
    )
    test_dataset = Cambridge(
        os.path.join("ShopFacade","testdata.txt"),
        "ShopFacade",
        anchors=None, #"ShopFacade/anchors.txt",
        shuffle=True,
        transform=composed,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )


    # === PRETRAINED FEATURE DETECTOR ===
    model = models.densenet161(pretrained=True)

    GAP = nn.AvgPool2d(7)

    dropout = nn.Dropout(p=0.6)

    net = Net(nAnchors).to(device)

    ################### TRAINING ####################
    startTime = time.time()
    # criterion = nn.CrossEntropyLoss()
    # klcriterion = nn.KLDivLoss()'

    mse = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    numEpochs = 25

    softmax = nn.Softmax(dim=1)

    scheduler = StepLR(optimizer, step_size=80, gamma=0.5)

    for epoch in range(numEpochs):  # loop over the dataset multiple times

        total_loss = 0
        train_accuracy = []
        train_distance = []
        median_train_distance = []
        train_dof_accuracy = []

        test_accuracy = []
        test_distance = []
        median_test_distance = []
        test_dof_accuracy = []

        temp_dist_test = []
        temp_dist_gt = []
        
        net.train()

        for i, data in enumerate(train_dataloader):
        
            if i%10 == 0:
                print(f"Time since start: {time.time()-startTime}")

            # get the inputs
            images, labels, dof, origxy = (
                data["image"],
                data["coordinates"],
                data["dof"],
                data["origxy"],
            )

            labels = labels.view(-1, 2*nAnchors)
            dof = dof.view(-1, 4)
            temp_class_labels = labels ** 2
            class_labels = torch.FloatTensor(temp_class_labels.size()[0], nAnchors)
            for j in range(class_labels.size()[0]):
                for k in range(class_labels.size()[1]):
                    class_labels[j][k] = (
                        temp_class_labels[j][2 * k] + temp_class_labels[j][2 * k + 1]
                    )
            
            images = images.to(device)
            labels = labels.to(device)
            dof = dof.to(device)
            class_labels = class_labels.to(device)

            temp_labels = labels
            temp_dof = dof
            images = Variable(images)
            labels = Variable(labels)
            class_labels = Variable(class_labels)
            dof = Variable(dof)
            

            optimizer.zero_grad()
            classify, regress, dof_regress = net(images)
            classify = softmax(classify)
            loss = custom_loss(classify, regress, labels, class_labels, dof, dof_regress, nAnchorpoints=nAnchors)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            _, predicted = torch.max(classify.data, 1)

            correct = 0
            correct_dof = 0
            diff = torch.abs(regress - labels)

            temp_dist = 0

            for j in range(predicted.size()[0]):
                if (
                    diff[j][2 * predicted[j]].data.cpu().numpy() ** 2
                    + diff[j][2 * predicted[j] + 1].data.cpu().numpy() ** 2
                ) ** 0.5 < 2:
                    correct += 1
                temp_dist += (
                    diff[j][2 * predicted[j]].data.cpu().numpy() ** 2
                    + diff[j][2 * predicted[j] + 1].data.cpu().numpy() ** 2
                ) ** 0.5
                median_train_distance.append(
                    (
                        diff[j][2 * predicted[j]].data.cpu().numpy() ** 2
                        + diff[j][2 * predicted[j] + 1].data.cpu().numpy() ** 2
                    )
                    ** 0.5
                )

            dof_distances = torch.abs(dof_regress - dof)
            predicted_dof, _ = torch.max(dof_distances.data, 1)

            for j in range(predicted_dof.size()[0]):
                if predicted_dof[j] < 0.3:
                    correct_dof += 1

            train_dof_accuracy.append(float(correct_dof) / predicted_dof.size()[0])
            train_accuracy.append(float(correct) / predicted.size()[0])
            train_distance.append(float(temp_dist) / predicted.size()[0])

        median_train_distance = np.array(median_train_distance)

        if (epoch+1) % 5 == 0:
            net.eval()

            for k, data in enumerate(test_dataloader):

                images, labels, dof, origxy = (
                    data["image"],
                    data["coordinates"],
                    data["dof"],
                    data["origxy"],
                )
                labels = labels.view(-1, 2*nAnchors)
                dof = dof.view(-1, 4)

                images = images.to(device)
                labels = labels.to(device)
                dof = dof.to(device)

                images = Variable(images)
                labels = Variable(labels)
                dof = Variable(dof)

                classify, regress, dof_regress = net(images)
                classify = softmax(classify)

                _, predicted = torch.max(classify.data, 1)

                correct = 0
                correct_dof = 0
                diff = torch.abs(regress - labels)

                for j in range(predicted.size()[0]):
                    if (
                        diff[j][2 * predicted[j]].data.cpu().numpy() ** 2
                        + diff[j][2 * predicted[j] + 1].data.cpu().numpy() ** 2
                    ) ** 0.5 < 2:
                        correct += 1
                    temp_dist_test.append(
                        [
                            diff[j][2 * predicted[j]].data.cpu().numpy()
                            + anchors[1][predicted[j]][0],
                            diff[j][2 * predicted[j] + 1].data.cpu().numpy()
                            + anchors[1][predicted[j]][1],
                        ]
                    )
                    temp_dist_gt.append([origxy[j].numpy()])
                    median_test_distance.append(
                        (
                            diff[j][2 * predicted[j]].data.cpu().numpy() ** 2
                            + diff[j][2 * predicted[j] + 1].data.cpu().numpy() ** 2
                        )
                        ** 0.5
                    )

                dof_distances = torch.abs(dof_regress - dof)
                predicted_dof, _ = torch.max(dof_distances.data, 1)

                for j in range(predicted_dof.size()[0]):
                    if predicted_dof[j] < 0.3:
                        correct_dof += 1

                test_dof_accuracy.append(float(correct_dof) / predicted_dof.size()[0])
                test_accuracy.append(float(correct) / predicted.size()[0])
                test_distance.append(float(temp_dist) / predicted.size()[0])

            test_errors = median_test_distance
            median_test_distance = np.array(median_test_distance)

            test_errors = [list(k) for k in np.array([test_errors])]
            test_errors = list(test_errors)

            """
            Saving test distances and ground truths for x,y for calculating test accuracy

            
            
            with open(os.path.join("ShopFacade","test_errors.txt"), "w") as f:
                pickle.dump(test_errors, f)

            with open(os.path.join("ShopFacade","pred_dist.txt"), "w") as f:
                pickle.dump(temp_dist_test, f)

            with open(os.path.join("ShopFacade","gt_dist.txt"), "w") as f:
                pickle.dump(temp_dist_gt, f)
            """

            np.savetxt(os.path.join("ShopFacade","run-22-09-20",f"test_errors_epoch{epoch}.txt"), np.squeeze(np.array(test_errors)))
            np.savetxt(os.path.join("ShopFacade","run-22-09-20",f"pred_dist_epoch{epoch}.txt"), np.squeeze(np.array(temp_dist_test)))
            np.savetxt(os.path.join("ShopFacade","run-22-09-20",f"gt_dist_epoch{epoch}.txt"), np.squeeze(np.array(temp_dist_gt)))
        
        if epoch%100+1 == 0:
            torch.save(net, os.path.join("ShopFacade", "nets", f"trainedNet_epoch{epoch}.pt"))

        print (
            "Epoch: %d, Training Loss: %.4f, Training Acc: %.4f , Train Mean Dist: %.4f , Train DOF Acc: %.4f, Train Median Dist: %.4f, Time: %.4f"
            % (
                epoch + 1,
                total_loss,
                (sum(train_accuracy) / float(len(train_accuracy))),
                (sum(train_distance) / float(len(train_distance))),
                (sum(train_dof_accuracy) / float(len(train_dof_accuracy))),
                np.median(median_train_distance),
                time.time() - startTime,
            )
        )

    torch.save(net, os.path.join(os.path.join("ShopFacade", "nets","CE0SQD1MSE1-ADAMOPT0-0003.pt")))