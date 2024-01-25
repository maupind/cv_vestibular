import os
import cv2
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

NUM_WORKERS = os.cpu_count()



class VideoDataset(Dataset):
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.video_files = [file for file in os.listdir(video_dir) if file.endswith('.mp4')]

        self.labels = [self.extract_label(file) for file in self.video_files]

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index):
        video_file= self.video_files[index]
        label = self.labels[index]

        video_frames = self.load_video_frames(os.path.join(self.video_dir, video_file))

        # Convert NumPy arrays to PyTorch tensors
        video_frames = [torch.from_numpy(frame) for frame in video_frames]

        # Stack the frames along a new dimension to create a 4D tensor (sequence of frames)
        video_frames = torch.stack(video_frames, dim=0)

        # Convert label to a tuple of PyTorch tensors
        label = (torch.tensor(label[0]), torch.tensor(label[1]))
        
        return video_frames, (label[0], label[1])   

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
    
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames


def create_dataloaders(
    data_dir: str,
    transform: transforms.Compose,
    test_size: int,
    batch_size: int,
    num_workers: int=NUM_WORKERS
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
    video_dataset=VideoDataset(video_dir=data_dir)

    X = video_dataset

    y = [label[2] for label in video_dataset.labels]

    # Split into test/train
    X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=test_size, 
    random_state=33, 
    stratify = y)

   

    train_dataloader = DataLoader(
        X_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        X_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader




