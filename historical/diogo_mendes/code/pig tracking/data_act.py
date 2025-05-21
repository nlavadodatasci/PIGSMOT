import pandas as pd
import torch, torchvision
import torchvision.tv_tensors
import os, json

behaviors = ('EAT', 'EXPLORATION', 'SOCIAL INTERACTION',
    'HEAD BODY KNOCK', 'LYING', 'NOSING', 'INTERACTION WITH OBJECTS',
    'MOUNT ATTEMPT', 'BLOCK OF FEEDING', 'BITE', 'MOUNT SEXUAL ACTION',
    'INACTIVE', 'ABNORMAL BEHAVIOUR', 'FLEE', 'CHASE', 'THREAT',
    'PRESSING PARALLEL / INVERSE')

violent_behaviors = ('BITE', 'HEAD BODY KNOCK', 'NOSING', 'PRESSING PARALLEL / INVERSE', 'BLOCK OF FEEDING')
violent_behaviors = torch.tensor([behaviors.index(b) for b in violent_behaviors])

#### o load não está a ser feito corretamente devido ao total frames, usar o numero obtido nos frames####
frames_2min=[0,3599,7199,  10799,  14399, 17999,21599, 25199,  28799, 32399,  35999, 39599,  43199, 46799, 50399,  53999, 57599, 61199,  64799,  68399,  71999,  165599, 172799,  176399,  179999,  183599, 187199,  190799,  194399,  197999,  201599,  205199,  208799,  212399,  215999,  219599,  223199,  226799,  230399,  233999,  237599,  241199,  244799,  248399,  251999,  255599,  259199,  262799,  266399,  269999,  273599,  277199, 280799,  284399,  287999,  291599,  295199,  298799,  302399,  305999,  309599,  313199,  316799,  320399, 323999, 327599,  331199, 334799,  338399]
frames_30s=[239399, 240299,  241199,  242099,  242999,  243899,  244799,  245699,  246599,  247499,  248399,  249299,  250199,  251099,  251999,  252899,  253799]
## os valores equivalem ao valor do frame 
#fazer calculo de (valor+1)/30 a partir do segundo frame para obter o segundo que ocorre a ação
# adaptar o codigo para apenas pegar naquele segundo e assumir a ação, nao precisamos do stop pq so temos a imagem quer seja ponto ou continuo é =




def load_activities(filename, fps, total_frames, npigs=10):
    df = pd.read_excel(filename, 'Total')
    
    labels = torch.zeros((total_frames, npigs, len(behaviors)), dtype=int)
    for _, row in df.iterrows():
        if pd.isna(row['Subject']):
            continue
        start_frame = int(round(row['Time']*fps))  # convert to frame
        if row['Status'] == 'STOP':
            continue
        if row['Status'] == 'START':
            stop_rows = df.loc[(df['Status'] == 'STOP') & (df['Time'] > row['Time']) & (df['Behavior'] == row['Behavior']) & (df['Subject'] == row['Subject'])]
            if len(stop_rows) > 0:
                stop_row = stop_rows.iloc[0]
                end_frame = int(round(stop_row['Time']*fps))
            else:  # no stop row ? weird, assume behavior stops at the end of the video.
                end_frame = total_frames-1
        if row['Status'] == 'POINT':
            end_frame = start_frame + int(round(3*fps))
        label = behaviors.index(row['Behavior'])
        pig = int(row['Subject'].split()[0][1:])-1
        labels[start_frame:end_frame+1, pig, label] = 1
    return labels

def to_binary_activities(labels):
    return labels[:, :, violent_behaviors].any(2)

class Pigs:
    def __init__(self, transform,npigs):
        self.images = sorted(os.listdir('data_pigs_all/images'))
        self.transform = transform
        self.labels_2mins = to_binary_activities(load_activities('Observação Pen A.xlsx', 1/120, 300))   # verificar esta função acho que os fps e total frames estão mal não devia ser 3600, 300???
        self.labels_30s = to_binary_activities(load_activities('Observação Pen A.xlsx', 1/30, 300))    # verificar esta função acho que os fps e total frames estão mal
        self.npigs = npigs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image = self.images[ix]
        
        x = torchvision.io.read_image(os.path.join('data_pigs_all/images', image))
        d = json.load(open(os.path.join('data_pigs_all/labels', image[:-3] + 'json')))
        y_bboxes = torch.zeros(self.npigs, 4)
        for i in range(self.npigs):
            #print(d['imagePath'])
            #print(len(d['shapes']))        
            pts = d['shapes'][i]['points']
            if d['shapes'][i]['shape_type'] == 'rectangle':
                y_bboxes[i] = torch.tensor((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
            if d['shapes'][i]['shape_type'] == 'polygon':
                y_bboxes[i] = torch.tensor((min(pt[0] for pt in pts), min(pt[1] for pt in pts),
                    max(pt[0] for pt in pts), max(pt[1] for pt in pts)))
        y_bboxes = torchvision.tv_tensors.BoundingBoxes(y_bboxes, format='XYXY', canvas_size=x.shape[1:])
        if self.transform:
            x, y_bboxes = self.transform(x, y_bboxes)
        # normalize 0-1
        y_bboxes[:, [0, 2]] = y_bboxes[:, [0, 2]] / x.shape[2]  # alterado yt.canvas_size(1)
        y_bboxes[:, [1, 3]] = y_bboxes[:, [1, 3]] / x.shape[1]  # alterado yt.canvas_size(0)
        # xyxy -> cxcywh
        y_bboxes = torchvision.ops.box_convert(y_bboxes, 'xyxy', 'cxcywh')
        
        # activities 
        frame = int(image[:-4])-1
        if frame > 200:
            y_acts = self.labels_30s[frame]
        else:
            y_acts = self.labels_2mins[frame]
        
        return x, y_bboxes, y_acts




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ds = Pigs(None, 10)
    for i, (x, y_bboxes, y_acts) in enumerate(ds):
        plt.imshow(x.permute(1, 2, 0))
        for y_bbox, y_act in zip(y_bboxes, y_acts):
            c = 'red' if y_act == 1 else 'blue'
            y_bbox = y_bbox * torch.tensor((x.shape[2], x.shape[1], x.shape[2], x.shape[1]))
            plt.gca().add_patch(patches.Rectangle((y_bbox[0]-y_bbox[2]/2, y_bbox[1]-y_bbox[3]/2), y_bbox[2], y_bbox[3], linewidth=1, edgecolor=c, facecolor='none'))
        plt.title(f'frame={i}')
        plt.show()
