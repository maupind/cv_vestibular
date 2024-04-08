
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class VestibularNetwork(nn.Module):
    """
    Model architecture inspired by TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, num_classes: int, batch_size: int):
        super().__init__()
        self.video_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.BatchNorm3d(hidden_units),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm3d(hidden_units), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        #self.label_block = nn.SequenThe batch_size parameter is set to 1 by default, but it's also used as an argument in the method signature. This could lead to confusion about which value is actually being used for the batch size during the forward pass.tial(
        # nn.Linear(in_features=1, out_features=1),
        #  nn.ReLU()
        #)

        self.common_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_units),
            nn.ReLU(),
            nn.Conv3d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm3d(hidden_units),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, batch_first=True)
          
        self.classifier = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, video_frames, batch_size):
        # Process video frames
        video_frames = video_frames.float()
        #print(f"Shape of first video_frames{video_frames.shape}")
        (batch_size, sequence_length, channels, height, width) = video_frames.size()
        #print(f"Shape of second video_frames{video_frames.shape}")
        video_frames_reshaped = video_frames.view(batch_size * sequence_length, channels, height, width)

        #print(f"Shape of video_frames before permutation:{video_frames_reshaped.shape}")
        video_frames_reshaped = video_frames_reshaped.permute(1, 0, 2, 3)  
        #print(f"Shape of video_frames after permutation:{video_frames_reshaped.shape}")
      
       # Apply the convolutional layers
        print("start first layer")
        x = self.video_block(video_frames_reshaped)
        print("start second layer")
        x = self.common_block(x)
        x = torch.mean(x, dim=[2, 3])  # Pooling or flattening operation, adjust as needed

    # Reshape for LSTM input
        x = x.view(batch_size, sequence_length, -1)

    # Apply LSTM
        print("start lstm")
        lstm_out, _ = self.lstm(x)

    # Aggregate predictions using the final state of the LSTM
        lstm_out = lstm_out[:, -1, :]

    # Apply classifier
        print("classify")
        x = self.classifier(lstm_out)

        return x
















################### Initial trial with LSTM and labels ##########################3
      # x = self.video_block(video_frames_reshaped)
      #  print(f"block 1 {x.shape}")
        # Process labels
       # labels = labels.float()
       # print("Label dtype:", labels.dtype)
        #label_x = self.label_block(labels)
        #print("Size of label_x:", label_x.size())
        #label_x = label_x.unsqueeze(0).unsqueeze(1)  # Add singleton dimensions at the beginning
        #label_x = label_x.expand(batch_size, 1, 540, 960)  # Expand to the desired size
        # Concatenate the processed video frames and labels
        #print("Size of video_x:", video_x.size())
       # print("Size of label_x:", label_x.size())
        #print("Pre conatenate output of label x", label_x)
        #x = torch.cat((video_x, label_x), dim=1)
        #print("Post concatenate output", x)
        # Continue with the common block and classifier
      #  x = self.common_block(x)
      #  print(f"block 2 {x.shape}")
        # Flatten spatial dimensions
      #  x = torch.mean(x, dim=[2, 3])  # Pooling or flattening operation, adjust as needed
      #  print(f"After pooling/flattening: {x.shape}")

        # Apply classifier
        #x = self.classifier(x)
       # print(f"block3: {x.shape}")
        #x = x[:, -1, :]
       # lstm_out, _ = self.lstm(x)
        
        # Aggregate predictions using the final state of the LSTM
       # lstm_out = lstm_out[: -1, :]  # Take the last timestep's output
        #print(f"lstm {lstm_out.shape}")
       # print(f"lstm output {lstm_out}")
       # return x


       
