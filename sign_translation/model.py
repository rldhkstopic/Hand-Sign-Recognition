import torch.nn as nn
import torch.nn.functional as F

class gesture_fc_type_consonant(nn.Module):
    def __init__(self):
        super(gesture_fc_type_consonant, self).__init__()

        self.fc1 = nn.Linear(3*21, 500)
        self.fc2 = nn.Linear(500,128)
        self.fc3 = nn.Linear(128, 14)


    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1, 3*21)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x
    


class gesture_fc_type_vowel(nn.Module):
    def __init__(self):
        super(gesture_fc_type_vowel, self).__init__()

        self.fc1 = nn.Linear(3*21, 500)
        self.fc2 = nn.Linear(500,128)
        self.fc3 = nn.Linear(128, 17)


    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1, 3*21)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x
    

class rockpaper(nn.Module):
    def __init__(self):
        super(rockpaper, self).__init__()

        self.fc1 = nn.Linear(3*21, 500)
        self.fc2 = nn.Linear(500,128)
        self.fc3 = nn.Linear(128, 2)


    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1, 3*21)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x