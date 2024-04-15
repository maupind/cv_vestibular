import os
import cv2
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, RandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold


NUM_WORKERS = os.cpu_count()

class VideoDataset(Dataset):
    def __init__(self, video_dir, video_transform = None, clip_length = 300):
        super(VideoDataset).__init__()
        self.video_dir = video_dir
        self.video_files = [file for file in os.listdir(video_dir) if file.endswith('.mp4')]
        self.labels = [self.extract_label(file) for file in self.video_files]
        self.outcomes = [label[2] for label in self.labels]
        self.label_features = [self.extract_label(file)[:2] for file in self.video_files]  # Ignore outcome label
        self.video_transform = video_transform
        self.clip_length = clip_length

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index):
        video_file= self.video_files[index]
        label = self.labels[index]
        # Ensuring outcome is a 1D tensor
        outcome = torch.tensor(label[2])
        # Create a new tuple for label without modifying the original label
        label_without_outcome = (torch.tensor(label[0]), torch.tensor(label[1]))

        # Update the variable name to reflect the change
        label = label_without_outcome

        video_path = os.path.join(self.video_dir, video_file)

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Read frames dynamically
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1

            # Check if enough frames are read
            if frame_count >= self.clip_length:
                break

        # Release video capture object
        cap.release()

        # Apply transformations if specified
        if self.video_transform:
            frames = [self.video_transform(frame) for frame in frames]

        # Convert frames to tensor
        frames = torch.stack(frames)

        return frames, outcome

    def extract_label(self, filename):
        # Example: Extract labels using regular expression
        match = re.search(r'clip\d+_(\w+)_(\w+)_(\w+)', filename)
        if match:
            eye_side = 1 if match.group(1) == 'L' else 0
            test_pos = 1 if match.group(2) == 'LHPD' else 0
            outcome = 0 if match.group(3) == 'N' else 1
            return (eye_side, test_pos, outcome)
        else:
            return (-1, -1, -1)  # Default labels if not found
    
   # def load_video_frames(self, video_path):
    #    cap = cv2.VideoCapture(video_path)

     #   frames = []
      #  while cap.isOpened():
       #     ret, frame = cap.read()
        #    if not ret:
         #       break
          #  frames.append(frame)

        #cap.release()
        #return frames

    def adjust_clip_length(self, frames):
        # Adjust clip length by selecting a subset of frames
        if len(frames) > self.clip_length:
            start_idx = (len(frames) - self.clip_length) // 2
            frames = frames[start_idx:start_idx + self.clip_length]
        return frames

def create_dataloaders(
    data_dir: str,
    transform: transforms.Compose,
    test_size: int,
    batch_size: int,
    num_workers: int=NUM_WORKERS,
    random_seed: int = 33
):
    """
    Loads data using the VideoDataset class, splits into test/train, and turns them into a dataset and then dataloader

    Args:
        train_dir: path to training data
        test_dir: path to test data
        transform: the torchvision transforms to perform on the data
        batch_size: number of samples per batch in Dataloaders
        num_workers: int for number of workers per dataloader
    
    Returns:
        a tuple of (train_dataloader, test_dataloader, class_names)
        Class_names is a list of the target classes

    """
    # Load the data
    video_dataset=VideoDataset(video_dir=data_dir, video_transform = transform, clip_length=60)
    y = video_dataset.outcomes
    combined_data = (video_dataset.video_files, video_dataset.label_features)
    X = combined_data





    # Split into test/train
    train_idx, test_idx = train_test_split(np.arange(len(video_dataset)), 
    test_size=test_size, 
    random_state=33, 
    stratify = video_dataset.outcomes)
  
    train_dataset = Subset(video_dataset, train_idx)
    test_dataset = Subset(video_dataset, test_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, video_dataset



