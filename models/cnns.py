# %%
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN using convolutional layers
class CNN(nn.Module):
    def __init__(self, num_classes, input_channel):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Adaptive pooling layer to ensure the output size is (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Pooling layer
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        return x

class CNNNLayer(nn.Module):
    def __init__(self, num_classes, input_channel, n_layer):
        super(CNNNLayer, self).__init__()
        
        # Convolutional layers
        self.convs = nn.ModuleList()

        for i in range(n_layer):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, padding=1))
            else:
                self.convs.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        
        # Adaptive pooling layer to ensure the output size is (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU activations
        for conv in self.convs:
            x = F.relu(conv(x))
        
        # Pooling layer
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        return x

def CNN_3layer(num_classes, input_channel):
    return CNN(num_classes, input_channel)

def CNN_Nlayer(num_classes, input_channel, n_layer):
    return CNNNLayer(num_classes, input_channel, n_layer)