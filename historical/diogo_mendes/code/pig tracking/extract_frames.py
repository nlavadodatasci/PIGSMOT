import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--videos',default=r"")
parser.add_argument('--output_path', default="images_30s")
#parser.add_argument('--resize', action='store_true')
args = parser.parse_args()


from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
import imageio
import os

videos = sorted(os.listdir(args.videos))
videos = [os.path.join(args.videos, video) for video in videos]
start = 0
frame_number=0
total=0
for video in videos:
    #print(video)
    for i, frame in enumerate(imageio.imiter(video)):
        total+=1
        if total%900==0 or total==1:
            frame_number+=1
            #if args.resize:
                #frame = resize(frame, (224, 224), preserve_range=True).astype('uint8')
            imsave(os.path.join(args.output_path, f'{frame_number:06d}.jpg'), frame)
            print(f'"{frame_number:06d}": {start+i}, ', end='')
    start += i+1
    
print()
    #/data/bioeng/pigs/videos

#python -u
