
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class VestibularNetwork(nn.Module):
    def __init__(self, hidden_units: int, input_shape: int, output_shape: int, num_classes: int):
        super().__init__()

        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first layer to accept batches of images with different sizes
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify the last layer of ResNet to match the output size
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Use Identity to keep the features extracted by ResNet

        self.hidden_units = hidden_units

        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_ftrs, hidden_size=hidden_units, batch_first=True)
        
        # Classifier
        self.classifier = nn.Linear(hidden_units, 1)

    def forward(self, video_frames):
        batch_size, sequence_length, c, h, w = video_frames.size()
        # Reshape to treat the sequence of frames as a batch
        video_frames = video_frames.view(batch_size * sequence_length, c, h, w)
        
        # Extract features using ResNet
        features = self.resnet(video_frames)
        
        # Reshape features to separate the original batch and sequence
        features = features.view(batch_size, sequence_length, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Classifier
        output = self.classifier(lstm_out)
        
        return output


#class VestibularNetwork(nn.Module):
#    """
#    Model architecture inspired by TinyVGG from: 
#    https://poloclub.github.io/cnn-explainer/
#    """
#    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, num_classes: int, batch_size: int = 1):
#        super().__init__()
#        self.video_block = nn.Sequential(
#            nn.Conv2d(in_channels=input_shape, 
#                      out_channels=hidden_units, 
#                      kernel_size=3, # how big is the square that's going over the image?
#                      stride=1, # default
#                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
#            nn.BatchNorm2d(hidden_units),
#            nn.ReLU(),
#            nn.Conv2d(in_channels=hidden_units, 
#                      out_channels=hidden_units,
#                      kernel_size=3,
#                      stride=1,
#                      padding=1),
#            nn.BatchNorm2d(hidden_units), 
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2,
#                         stride=2) # default stride value is same as kernel_size
#        )
        #self.label_block = nn.SequenThe batch_size parameter is set to 1 by default, but it's also used as an argument in the method signature. This could lead to confusion about which value is actually being used for the batch size during the forward pass.tial(
         #   nn.Linear(in_features=1, out_features=1),
         #   nn.ReLU()
        #)

#        self.common_block = nn.Sequential(
#            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
#            nn.BatchNorm2d(hidden_units),
#            nn.ReLU(),
#            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
#            nn.BatchNorm2d(hidden_units),
#            nn.ReLU(),
#            nn.MaxPool2d(2)
#        )

        # Define LSTM layer
 #       self.lstm = nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, batch_first=True)
    

  #      self.classifier = nn.Linear(in_features=hidden_units, out_features=1)

   # def forward(self, video_frames, batch_size):
        # Process video frames
    #    video_frames = video_frames.float()
     #   (batch_size, sequence_length, channels, height, width) = video_frames.size()
      #  video_frames_reshaped = video_frames.view(batch_size*sequence_length, channels, height, width)
        #video_frames = video_frames.permute(2, 0, 1).unsqueeze(0)
        #video_frames = video_frames.repeat(batch_size, 1, 1, 1)
       # print("Video size:", video_frames_reshaped.shape)
       # x_list = []
       # for i in range(batch_size):
            # Process each batch separately
        #    x_batch = self.video_block(video_frames_reshaped[i * sequence_length:(i + 1) * sequence_length])
         #   x_batch = self.common_block(x_batch)
          #  x_batch = torch.mean(x_batch, dim=[2, 3])  # Pooling or flattening operation, adjust as needed
           # x_list.append(x_batch)

            # Concatenate the outputs of all batches
      #  x = torch.cat(x_list, dim=0)
      #  print(f"after cat: {x.shape}")

        # Reshape for LSTM input
       # x = x.view(batch_size, sequence_length, -1)

        # Apply LSTM
       # lstm_out, _ = self.lstm(x)

        # Aggregate predictions using the final state of the LSTM
       # lstm_out = lstm_out[:, -1, :]

        # Apply classifier
      #  x = self.classifier(lstm_out)
      #  print(f"classifier: {x.shape}")

      #  return x
















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


       
