import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class Resnet(nn.Module):
    def __init__(self, num_inputs, num_actions):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Resnet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(2048, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        
    def forward(self, inputs):
        """Extract feature vectors from input images."""
        inputs, (hx, cx) = inputs
        with torch.no_grad():
            features = self.resnet(inputs)
        
        features = features.view(-1, 2048)
        #features = features.reshape(features.size(0), -1)
        hx, cx = self.lstm(features, (hx, cx))
        x = hx
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        


