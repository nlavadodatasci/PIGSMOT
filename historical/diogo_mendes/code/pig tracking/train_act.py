import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--npigs', type=int, default=10)

args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
from time import time
import data_act, model_act
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 

############################### DATA ###############################

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
ds = data_act.Pigs(aug,args.npigs)    # verificar se é assim
#ds = torch.utils.data.Subset(ds, range(10))


i = int(0.8*len(ds))
tr = torch.utils.data.Subset(ds, range(0, i))
ts = torch.utils.data.Subset(ds, range(i, len(ds)))


# Criando DataLoaders para treino e teste
train = torch.utils.data.DataLoader(tr, 8, True, num_workers=0, pin_memory=True)
test = torch.utils.data.DataLoader(ts, 1, True, num_workers=0, pin_memory=True)



############################### MODEL ###############################

model = model_act.Detector(args.npigs)  # verificar se é assim
model.to(device)

############################### TRAIN ###############################
#print(device)

opt_slow = torch.optim.Adam(model.backbone.parameters(), 1e-4)
opt_fast = torch.optim.Adam(list(model.bboxes.parameters()) + list(model.acts.parameters()))


for epoch in range(0):  #args.epochs
    tic = time()
    avg_loss = 0
    avg_loss_acts=0
    for x, y_boxes, y_acts in train:
        
      
        x = x.to(device)
        
        y_boxes = y_boxes.to(device)  # (N, 10, 4)   # verificar se é assim
        y_acts=y_acts.to(device)                    # verificar se é assim
        
        pred_boxes, pred_acts = model(x)  # pred.shape = (N, 10, 4)

        loss = torch.nn.functional.l1_loss(pred_boxes, y_boxes)
        loss += torch.nn.functional.binary_cross_entropy(pred_acts, y_acts.float())
            #loss = torchvision.ops.generalized_box_iou_loss(torchvision.ops.box_convert(y, 'cxcywh', 'xyxy'), torchvision.ops.box_convert(pred, 'cxcywh', 'xyxy')).mean() + \
        opt_slow.zero_grad()
        opt_fast.zero_grad()
        loss.backward()
        opt_slow.step()
        opt_fast.step()
        avg_loss += float(loss)/len(ds)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {avg_loss} ')

#torch.save(model.cpu(), 'model500_act.pth')

############################### test ###############################


model = torch.load('model1000_act.pth', map_location=device)

############################### EVAL ###############################


import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import imageio
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import BinaryAccuracy
import os

map = MeanAveragePrecision('cxcywh', class_metrics=True)
acc = BinaryAccuracy()
frames = []
image_folder = 'frames'  # Pasta para salvar as imagens individuais
os.makedirs(image_folder, exist_ok=True)

print("acabou o treino")
for i, (x, y_boxes, y_acts) in enumerate(tqdm(test)):
    x = x.to(device)
        
    with torch.no_grad():
        ŷ_boxes, ŷ_acts = model(x)

    y_boxes = y_boxes[0].cpu() * torch.tensor(x.shape[2::][::-1]).repeat(2)[None]
    ŷ_boxes = ŷ_boxes[0].cpu() * torch.tensor(x.shape[2::][::-1]).repeat(2)[None]
    
    map.update(
        [{'boxes': ŷ_boxes, 'scores': torch.ones(args.npigs), 'labels': torch.arange(args.npigs)}],
        [{'boxes': y_boxes, 'labels': torch.arange(args.npigs)}])
    acc.update(ŷ_acts.cpu(), y_acts)
    x = x.cpu()
    
    plt.clf()
    plt.imshow(x[0].permute(1, 2, 0))
    
    ŷ_acts = torch.round(ŷ_acts) 
    ŷ_acts = torch.where(ŷ_acts == 1, True, False)
    ŷ_acts = ŷ_acts[0]
    
    for b, act in zip(ŷ_boxes, ŷ_acts):
        c = 'r' if act else 'b'
        plt.gca().add_patch(patches.Rectangle((b[0] - b[2]/2, b[1] - b[3]/2), b[2], b[3], linewidth=2, edgecolor=c, facecolor='none'))
    
    plt.title(f'Frame {i+1}')
    plt.draw()
    
    # Salvar como imagem JPEG
    image_path = os.path.join(image_folder, f'frame_{i:04d}.jpg')
    plt.savefig(image_path)
    
    # Converter para numpy array e adicionar aos frames para o GIF
    frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

map = map.compute()
acc = acc.compute()
print('mAP:', map)
print('Acc:', acc)

# Salvar o GIF animado
gif_path = 'model1000_act.gif'
imageio.mimsave(gif_path, frames, fps=3, duration=0.3, loop=0)

# Salvar resultados em um arquivo de texto
resultado_path = 'resultado1000_act.txt'
with open(resultado_path, 'w') as f:
    f.write(f'mAP: {map}\nAcc: {acc}\n')

print(f'Frames salvos em: {image_folder}')
print(f'GIF salvo em: {gif_path}')
print(f'Resultados salvos em: {resultado_path}')
