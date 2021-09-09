import argparse
import os
import re
import numpy as np
from tqdm import tqdm
import yaml
import shutil
import torch
import torchvision

from PIL import Image
from dataloaders import make_data_loader
from modeling.deeplab import *
import dataloaders.utils
from utils.image_sectioning import section_images
from utils.assemble_predictions import assemble_predictions

path_regex = re.compile('.+?/(.*)$')

def setup_argparser():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Prediction")
    parser.add_argument('--model', type=str, default=None, help='provide path to model')
    parser.add_argument('--dataset', type=str, default='marsh',
                        choices=['pascal', 'coco', 'cityscapes','marsh'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--dataset_path', type=str, default=None, help='provide path to dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for prediction (default: 2)')
    parser.add_argument('--workers', type=int, default=8,
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
    #args.crop_size = model_data['crop_size']
    return args

def setup_img_sectioning_params(args):
    re_fbase = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')
    section_dim = [4, 3]  # columns, rows to split input image into
    img_dim = [4096, 2160]

    patch_dim = [int(img_dim[0]/section_dim[0]), int(img_dim[1]/section_dim[1])]

    #avg_dim= int((patch_dim[0] + patch_dim[1])/2)
    args.crop_size = patch_dim

    pred_format = "{}\t{:4.3f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\t{:5.1f}\n"
    params = {'section_dim': section_dim, 'crop_size': patch_dim,  'fmt': pred_format, 're_fbase': re_fbase, 'workers': args.workers, 'write_imgs': True}

    #setup temporary folder to store section data
    tmp_folder = './tmp'  # start with a clean tmp folder
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder)
    else:
        os.mkdir(tmp_folder)
    params['outfld'] = tmp_folder

    return params, args

def make_predictions(model, dataloader, args):
    model.eval()
    if args.cuda:
        model.cuda()

    torch.no_grad()

    # process samples
    tbar = tqdm(dataloader, desc='predictions')
    for i, sample in enumerate(tbar):

        # unpackage sample
        image = sample['image']
        label = sample['label']
        dim = sample['dim']
        dim = torch.cat((dim[0].unsqueeze(1), dim[1].unsqueeze(1)), dim=1)

        if args.cuda:
            image = image.cuda()

        # forward pass through model and make predictions
        output = model(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        pred_rgb = dataloaders.utils.encode_seg_map_sequence(pred, args.dataset,
                                                             dtype='int')  # convert predictions to rgb masks

        # write out masks (resize to original image dimensions)
        for lbl, d, mask in zip(label, dim, pred_rgb):
            w = d[0]
            h = d[1]
            mask = torchvision.transforms.ToPILImage()(mask)
            mask = mask.resize((w, h), Image.NEAREST)

            base_path, fn = os.path.split(lbl)
            base_fn, ext = os.path.splitext(fn)
            outdir = base_path + "/preds/"
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            outpath = outdir + base_fn + "_mask.png"
            mask.save(outpath)

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    args.train = False
    args = parse_model_configfile(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #setup dataset
    params, args = setup_img_sectioning_params(args)
    print('sectioning images for prediction')
    section_data = section_images(args.dataset_path, params)
    #print(section_data)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    m = path_regex.findall(args.dataset_path)
    dirpath_sub = m[0]
    root_dir = os.path.join(params['outfld'], dirpath_sub)
    print('dirpath_sub {}'.format(dirpath_sub ))
    dataloader, nclass = make_data_loader(args, root=root_dir, **kwargs)

    # setup model
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    #make predictions for dataset using model
    make_predictions(model, dataloader, args)

    print('assembling predicted images')
    assemble_predictions(section_data, params)


if __name__ == "__main__":
   main()