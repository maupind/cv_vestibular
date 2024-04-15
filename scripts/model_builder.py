
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


    

class VestibularNetwork(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, num_classes: int, batch_size: int):
        super(VestibularNetwork, self).__init__()
        self.video_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.BatchNorm3d(hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=hidden_units, 
                     out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm3d(128), 
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=(1, 2, 2),
                         stride=(1, 2, 2)) # default stride value is same as kernel_size
        )
        #self.label_block = nn.SequenThe batch_size parameter is set to 1 by default, but it's also used as an argument in the method signature. This could lead to confusion about which value is actually being used for the batch size during the forward pass.tial(
        # nn.Linear(in_features=1, out_features=1),
        #  nn.ReLU()
        #)

        self.common_block = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=256, 
                     out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm3d(64), 
            #nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=2,
                         stride=2)
        )

     
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=389376, hidden_size=128, batch_first=True)

        # Define linear layers after LSTM
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(p=0.2)      
        self.fc2 = nn.Linear(64, 16)
        #self.dropout2 = nn.Dropout(p=0.2)
        #self.fc3 = nn.Linear(32, 16)          
        self.classifier = nn.Linear(in_features=16, out_features=1)

    def forward(self, video_frames, batch_size):
        # Process video frames
        video_frames = video_frames.float()
        #print(f"Shape of first video_frames{video_frames.shape}")
        (batch_size, sequence_length, channels, height, width) = video_frames.size()
        #print(f"Shape of second video_frames{video_frames.shape}")
        video_frames_reshaped = video_frames.view(batch_size, sequence_length, channels, height, width)

        #print(f"Shape of video_frames before permutation:{video_frames_reshaped.shape}")
        video_frames_reshaped = video_frames_reshaped.permute(0, 2, 1, 3, 4)
        #print(f"Shape of video_frames after permutation:{video_frames_reshaped.shape}")
      
       # Apply the convolutional layers
        print("start first layer")
        x = self.video_block(video_frames_reshaped)
        print("start second layer")
        x = self.common_block(x)
        print(f"size after common {x.shape}")
        #x = torch.mean(x, dim=[2, 3])  # Pooling or flattening operation, adjust as needed
        print(f"size before lstm{x.shape}")
    # Reshape for LSTM input
        x = x.view(1, 30, 389376)

    # Apply LSTM
        
        print("start lstm")
        lstm_out, _ = self.lstm(x)

    # Aggregate predictions using the final state of the LSTM
        x = lstm_out[:, -1, :]


        # Apply linear layers
        print(f"L layer 1")
        
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        print(f"L layer 2")
        x = torch.relu(self.fc2(x))
       # x = self.dropout2(x)        
      #  x = torch.relu(self.fc3(x))
    
      
    # Apply classifier
        print("classify")
        x = self.classifier(x)

        return x

















       