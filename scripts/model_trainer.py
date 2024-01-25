import sys
sys.path.append("/home/danny/Documents/Projects/cv_vestibular")
from data_loader import create_dataloaders
from model_builder import VestibularNetwork
import torch


model = VestibularNetwork(input_shape = 3,
                          hidden_units = 10,
                          output_shape = 1,
                          num_classes=2)


train_dataloader, test_dataloader = create_dataloaders(
    data_dir="/path/to/data/dir",
    transform=None,
    test_size=0.5,
    batch_size=1
)

############################## Trial for a random pass ############################

#batch = next(iter(train_dataloader))

#video_frames, labels = batch 

# Infer shapes from the tensors
#video_frames_shape = video_frames[0].shape[1:]
#labels_shape = labels[0].shape[0:]

#random_video_frames = torch.randint(0, 256, video_frames_shape, dtype=torch.uint8)
#random_labels = torch.randint(0, 2, labels_shape, dtype=torch.long)


#model.eval()
#with torch.inference_mode():
#    pred=model(random_video_frames, random_labels)
