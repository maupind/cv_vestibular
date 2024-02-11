
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class VestibularNetwork(nn.Module):
    """
    Model architecture inspired by TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, num_classes: int, batch_size: int = 1):
        super().__init__()
        self.video_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        #self.label_block = nn.Sequential(
         #   nn.Linear(in_features=1, out_features=1),
         #   nn.ReLU()
        #)

        self.common_block = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, batch_first=True)
    

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    
    def forward(self, video_frames = None, batch_size: int = 1):
        if video_frames is None:
        # If video_frames is not provided, use the default training data tensor (X_train_tensor)
            video_frames = X_train_tensor
        # Process video frames
        video_frames = video_frames.float()
        batch_size, sequence_length, channels, height, width = video_frames.size()
        video_frames_reshaped = video_frames.reshape(batch_size * sequence_length, channels, height, width)
        #video_frames = video_frames.permute(2, 0, 1).unsqueeze(0)
        #video_frames = video_frames.repeat(batch_size, 1, 1, 1)
        print("Video size:", video_frames_reshaped.shape)

        x = self.video_block(video_frames_reshaped)
        print(f"block1: {x.shape}")
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
        x = self.common_block(x)
        print(f"block2: {x.shape}")
         # Apply LSTM
        x = x.view(batch_size, sequence_length, -1)  # Reshape for LSTM
        lstm_out, _ = self.lstm(x)
        
        # Aggregate predictions using the final hidden state of the LSTM
        lstm_out = lstm_out[:, -1, :]  # Take the last timestep's output
        print(f"lstm {lstm_out.shape}")
        x = self.classifier(lstm_out)
        print(f"block3: {x.shape}")
        return x


       
