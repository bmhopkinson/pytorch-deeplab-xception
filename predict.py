import argparse
import os
import numpy as np
from tqdm import tqdm
import yaml
import torch
import torchvision

from dataloaders import make_data_loader
from modeling.deeplab import *
import dataloaders.utils


def setup_argparser():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Prediction")
    parser.add_argument('--model', type=str, default=None, help='provide path to model')
    parser.add_argument('--dataset', type=str, default='marsh',
                        choices=['pascal', 'coco', 'cityscapes','marsh'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--dataset_path', type=str, default=None, help='provide path to dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for prediction (default: 2)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    return parser

def parse_model_configfile(args):
    model_data_path = os.path.join(args.model, 'model_data.yaml')
    model_data = yaml.safe_load( open(model_data_path,'r') )
    args.backbone = model_data['backbone']
    args.out_stride = model_data['out_stride']
    args.model_path = os.path.join(args.model, model_data['model_path'])
    args.crop_size = model_data['crop_size']
    return args

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    args.train = False
    args = parse_model_configfile(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    dataloader, nclass = make_data_loader(args, **kwargs)

    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # if args.cuda:
    #     model.module.load_state_dict(checkpoint['state_dict'])
    # else:
    #     model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    if args.cuda:
        model.cuda()
 #   tbar = tqdm(dataloader, desc='\r')
    torch.no_grad()

    for i, sample in enumerate(dataloader):
        print('starting batch {}'.format(i))
        image = sample['image']
        label = sample['label']
        if args.cuda:
            image = image.cuda()

        output = model(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        print('finished batch {}'.format(i))

        pred_rgb = dataloaders.utils.encode_seg_map_sequence(pred, args.dataset, dtype='int' )
        print('pred_rgb.size() {}'.format(pred_rgb.size()))

        for lbl, mask in zip(label, pred_rgb):
            print('mask.size() {}'.format(mask.size()))
            mask = torchvision.transforms.ToPILImage()(mask)
            base_path, fn = os.path.split(lbl)
            base_fn, ext = os.path.splitext(fn)
           # print('image size {}'.format(samp['dim']))
            outdir = base_path +"/preds/"
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            outpath = outdir + base_fn + "_mask.png"
            mask.save(outpath)

if __name__ == "__main__":
   main()