import os
import torch
import torch.utils.data as data
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import transforms
from data_loader import get_segmentation_dataset_video
from data_loader import CitySegmentation_video
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete, get_mask

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # output folder
        self.outdir = 'test_result'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        val_dataset = get_segmentation_dataset_video(args.dataset, split='demovideo', mode='testval',
                                               transform=input_transform)
        cityscape = CitySegmentation_video()
        self.img_roots, self.img_names = cityscape.get_image_infos()
        # for idx, item in enumerate(self.img_names):
        #     self.img_names[idx] = item.replace('.png', '')
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        # create network
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=True, root=args.save_folder).to(args.device)
        print('Finished loading model!')

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.model.eval()
        for i, image in enumerate(self.val_loader):
            image = image.to(self.args.device)
            outputs = self.model(image)

            pred = torch.argmax(outputs[0],1)
            pred = pred.cpu().data.numpy()
            predict = pred.squeeze(0)

            mask = get_color_pallete(predict, self.args.dataset)
            #mask.save(os.path.join(self.outdir, 'seg_{}.png'.format(i))) #save segmentation

            mask_np = np.asarray(mask)
            mask_np = get_mask(mask_np, 0)
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

            img = cv2.imread(self.img_roots[i])
            cv2.imwrite(os.path.join(self.outdir, 'msk_{}.png'.format(i)),cv2.add(mask_np,img))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            
            print('{} is completed'.format(i))
            if i==5:
                break
    

if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    evaluator.eval()