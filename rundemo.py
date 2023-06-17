import os

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from libs.models import get_model
from albumentations import (
    Compose,
    Normalize,
    Resize
)
from albumentations.pytorch import ToTensorV2

from utils.visualize import visualize, reverse_normalize
from libs.dataset import get_dataloader
from libs.loss_fn import get_criterion
from libs.helper_bedsrnet import do_one_iteration
from libs.device import get_device
from pytorch_grad_cam import GradCAM
from torchvision import datasets, transforms

def convert_show_image(tensor, idx=None):
    if tensor.shape[1] == 3:
        img = reverse_normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif tensor.shape[1] == 1:
        img = tensor * 0.5 + 0.5

    if idx is not None:
        img = (img[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img = (img.squeeze(axis=0).transpose(1, 2, 0) * 255).astype(np.uint8)

    return img


def main() -> None:
    test_transform = Compose([Resize(1024, 768), Normalize(mean=(0.5, ), std=(0.5, )), ToTensorV2()])
    test_loader = get_dataloader(
        "Jung",
        "bedsrnet",
        "test",
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        transform=test_transform,
    )

    # cpu or cuda
    device = get_device(allow_only_gpu=False)
    benet = get_model('cam_benet', in_channels=3, pretrained=True)
    benet.model = benet.model.to(device)
    srnet = get_model('srnet', pretrained=True)

    generator, discriminator = srnet[0].to(torch.device('cpu')), srnet[1].to(torch.device('cpu'))
    generator.eval()
    discriminator.eval()
    generator.to(device)
    discriminator.to(device)
    criterion = get_criterion("GAN", device)
    lambda_dict = {"lambda1": 1.0, "lambda2": 0.01}

    # target_layers = [benet.target_layer]
    # grad_cam = GradCAM(model=benet, target_layers=target_layers, use_cuda=(True if device == "cuda" else False))

    gts = []
    preds = []
    attmaps = []
    bgcolors = []
    psnrs = []
    ssims = []
    index = 0
    sampleSize = 4  #  len(test_loader)
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            if i >= sampleSize:
                break
            print(sample["img_path"][0])
            _, _, _, input, gt, pred, attention_map, back_ground, psnr, ssim = do_one_iteration(False,
                                                                                                sample,
                                                                                                generator,
                                                                                                discriminator,
                                                                                                benet,
                                                                                                None,
                                                                                                criterion,
                                                                                                device,
                                                                                                "evaluate",
                                                                                                lambda_dict)
            gts += list(gt)
            preds += list(pred)
            attmaps += list(attention_map)
            bgcolors += list(back_ground)
            psnrs.append(psnr)
            ssims.append(ssim)

    print(f"psnr: {np.mean(psnrs)}")
    print(f"ssim: {np.mean(ssims)}")


    figure = plt.figure(figsize=(9 * 3, 2 * 3 * sampleSize))  #  len(test_loader)))

    for idx, sample in enumerate(test_loader):
        if idx >= sampleSize:
            break
        img_path = sample['img_path'][0].split('/')[-1]

        plt.subplot(sampleSize, 5, idx * 5 + 1)
        plt.title(img_path + ' input image')
        plt.imshow(convert_show_image(sample["img"].clone().detach().cpu().numpy()))

        plt.subplot(sampleSize, 5, idx * 5 + 2)
        plt.title(img_path + ' Ground-Truth image')
        plt.imshow(convert_show_image(np.array(gts), idx=idx))

        plt.subplot(sampleSize, 5, idx * 5 + 3)
        plt.title(img_path + ' shadow removal image')
        plt.imshow(convert_show_image(np.array(preds), idx=idx))

        plt.subplot(sampleSize, 5, idx * 5 + 4)
        plt.title(img_path + ' back ground color image')
        plt.imshow(convert_show_image(np.array(bgcolors), idx=idx))

        plt.subplot(sampleSize, 5, idx * 5 + 5)
        plt.title(img_path + ' attention map')
        plt.imshow(convert_show_image(np.array(attmaps), idx=idx), cmap='jet', alpha=0.5)
        plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()
