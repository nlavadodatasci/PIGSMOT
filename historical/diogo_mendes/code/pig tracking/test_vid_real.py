import argparse
import os
from skimage.transform import resize
from tqdm import tqdm
import imageio
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import torchvision
matplotlib.use('Agg')
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.transforms import v2
 


parser = argparse.ArgumentParser()
parser.add_argument('--npigs', type=int, default=10)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('model200_act.pth', map_location=device)
model.to(device)
model.eval()

aug = v2.Compose([
    v2.Resize((512, 512)),
    #v2.RandomCrop((224, 224)),
    #v2.Resize((int(224*1.05), int(224*1.05))),
    #v2.RandomCrop((224, 224)),
    # is hflips and vflips a good idea? - food always in the same place
    #v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    #v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
])

def process_frame(frame, frame_number, output_folder):
    x = aug(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        #ŷ_boxes = model(x)
        ŷ_boxes,ŷ_acts = model(x)
    
    
    x=x.cpu()
    plt.clf()
    plt.imshow(x[0].permute(1, 2, 0))
    ŷ_boxes = ŷ_boxes[0].cpu() * torch.tensor(x.shape[2::][::-1]).repeat(2)[None]
    ŷ_acts=torch.round(ŷ_acts) 
    ŷ_acts = torch.where(ŷ_acts == 1, True, False)
    ŷ_acts=ŷ_acts[0]
    

    for b,act in zip(ŷ_boxes,ŷ_acts):
        #c = 'r' if act else 'b'  
        c='b'  
        plt.gca().add_patch(patches.Rectangle((b[0] - b[2]/2, b[1] - b[3]/2), b[2], b[3], linewidth=2, edgecolor=c, facecolor='none'))  #mudar a cor
        
        
        
    
    
    plt.axis('off')
    plt.draw()

    # Salvar a imagem individualmente
    #output_image_path = os.path.join(output_folder, f'image_{frame_number+1}.png')
    #plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    canvas = plt.gcf().canvas
    canvas.draw()
    frame_array = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    frame_array = frame_array.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Discard the alpha channel
    
    # Add the frame to the video writer
    video_writer.append_data(frame_array)


output_video_path = "tracking200_penA_manha.mp4"
    
    


#output_folder = 'output_images_200'
#os.makedirs(output_folder, exist_ok=True)

video_path = ""

# Get the number of frames in the video
video_reader = imageio.get_reader(video_path)
num_frames = video_reader.count_frames()

# Initialize video writer
video_writer = imageio.get_writer(output_video_path, fps=30)

# Process each frame with tqdm progress bar
for frame_number, frame in enumerate(tqdm(video_reader, total=num_frames)):
    frame = resize(frame, (512, 512), preserve_range=True).astype('uint8')
    frame = torch.from_numpy(frame).permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)
    process_frame(frame, frame_number, video_writer)

# Close the video writer
video_writer.close()
