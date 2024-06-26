import sys
sys.path.append("/home/danny/Documents/Projects/cv_vestibular")
from data_loader import create_dataloaders
from model_builder import VestibularNetwork
import torchvision
from torchvision import transforms
import torch


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(314, 314)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0,0,0], std=[1,1,1])
])




train_dataloader, test_dataloader, video_dataset = create_dataloaders(
    data_dir="/home/danny/Documents/hpd_clips_test",
    transform=transform,
    test_size=0.2,
    batch_size=1
)




def get_input_shape(dataloader):
    for batch_idx, (video_frames_batch, outcomes) in enumerate(dataloader):
        input_shape = video_frames_batch.shape
        return input_shape

# Call the function to get the input shape
input_shape = get_input_shape(train_dataloader)[2]



vest_model = VestibularNetwork(input_shape=3,
                          batch_size=1,
                          #out_channels=64,
                          hidden_units=64,
                          #layers=4,
                          #output_shape=1,
                          num_classes=1)

############################## Trial for a random pass ############################

#batch = next(iter(train_dataloader))

#video_frames, labels = batch

# Infer shapes from the tensors
#video_frames_shape = video_frames[0].shape[1:]
#labels_shape = labels[0].shape[0:]

#random_video_frames = torch.randint(0, 256, input_shape, dtype=torch.uint8)
#random_labels = torch.randint(0, 2, labels_shape, dtype=torch.long)


#vest_model.eval()
#with torch.inference_mode():
 #   pred=vest_model(random_video_frames)
for batch_idx, (video_frames, outcomes) in enumerate(train_dataloader):
    # Print the data in the current batch
    print("Batch Index:", batch_idx)
    print("Video Frames Shape:", video_frames.shape)
    print("Outcomes:", outcomes)