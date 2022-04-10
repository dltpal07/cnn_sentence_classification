import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(2, 1000), stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 1000), stride=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 1000), stride=1)

        self.pool1 = nn.MaxPool2d((5, 1), stride=(2, 1))
        self.pool2 = nn.MaxPool2d((4, 1), stride=(2, 1))
        self.pool3 = nn.MaxPool2d((3, 1), stride=(2, 1))

        self.fc1 = nn.Linear(300*8*1, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.bn = nn.BatchNorm2d(100)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        feature_vector = x.view(x.size(0), x.size(1), -1)
        feature_vector = feature_vector.unsqueeze(1)
        cv1 = self.pool1(self.relu(self.bn(self.conv1(feature_vector))))
        cv2 = self.pool2(self.relu(self.bn(self.conv2(feature_vector))))
        cv3 = self.pool3(self.relu(self.bn(self.conv3(feature_vector))))
        feature_vector = torch.cat((cv1, cv2, cv3), 1)
        feature_vector = feature_vector.view(feature_vector.size(0), -1)
        logits = self.fc2(self.relu(self.fc1(feature_vector)))
        return logits
