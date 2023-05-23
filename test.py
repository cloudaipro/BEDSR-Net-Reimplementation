import cv2
import argparse
import os
import torch
import torch.optim as optim
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from libs.device import get_device
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper_bedsrnet import infer
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed
import matplotlib.pyplot as plt
import numpy as np
from utils.visualize import visualize, reverse_normalize
from torchvision import transforms
def convert_show_image(tensor, idx=None):
    if tensor.shape[1]==3:
        img = reverse_normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif tensor.shape[1]==1:
        img = tensor*0.5+0.5

    if idx is not None:
        img = (img[idx].transpose(1, 2, 0)*255).astype(np.uint8)
    else:
        img = (img.squeeze(axis=0).transpose(1, 2, 0)*255).astype(np.uint8)

    return img

result_path = './shadow_removal_image.jpg'
image = cv2.imread("./IMG_4750.JPG")   #  "./rect_5.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape
m_h = 1024
m_w = 768
test_transform = Compose([Resize(m_h, m_w), Normalize(mean=(0.5, 0.5,0.5), std=(0.5, 0.5,0.5)), ToTensorV2()])

device = get_device(allow_only_gpu=False)
tensor = test_transform(image=image)
tensor = tensor['image'].unsqueeze(0).to(device)

benet = get_model('cam_benet', in_channels=3, pretrained=True)
benet.model = benet.model.to(device)
srnet = get_model('srnet', pretrained=True)
generator, discriminator = srnet[0].to(torch.device(device)), srnet[1].to(torch.device(device))
generator.eval()
discriminator.eval()
generator.to(device)
discriminator.to(device)
criterion = get_criterion("GAN", device)
lambda_dict = {"lambda1": 1.0, "lambda2": 0.01}
with torch.no_grad():

    with torch.set_grad_enabled(True):
        color, attmap, _ = benet(tensor)
        attmap = (attmap-0.5)/0.5
        back_color = torch.repeat_interleave(color.detach(), m_h*m_w, dim=0)
        back_ground = back_color.reshape(1, c, m_h, m_w).to(device)

    input = torch.cat([tensor, attmap, back_ground], dim=1)

    tensor = tensor.detach().cpu()
    attmap = attmap.detach().cpu()
    back_ground = back_ground.detach().cpu()
    shadow_removal_image = generator(input).detach().cpu()

figure = plt.figure(figsize = (9*3, 2*3))
plt.subplot(1, 4, 1)
plt.title('input image')
plt.imshow(image)   #  convert_show_image(tensor.clone().detach().cpu().numpy()))
plt.subplot(1, 4, 2)
plt.title('shadow removal image')
remove_original = shadow_removal_image.clone().detach().cpu().numpy()
removal = convert_show_image(shadow_removal_image.clone().detach().cpu().numpy())
plt.imshow(removal)
plt.subplot(1, 4, 3)
plt.title('back ground color image')
plt.imshow(convert_show_image(back_ground.clone().detach().cpu().numpy()))
plt.subplot(1, 4, 4)
plt.title('attention map')
plt.imshow(convert_show_image(attmap.clone().detach().cpu().numpy()), cmap='jet', alpha=0.5)
plt.colorbar()
plt.show()

if cv2.imwrite(result_path, cv2.cvtColor(removal, cv2.COLOR_RGB2BGR)):
  print("saved")