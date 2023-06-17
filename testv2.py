import cv2
import os
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from libs.device import get_device
import matplotlib.pyplot as plt
import numpy as np
from utils.visualize import visualize, reverse_normalize
from libs.models import models
from libs.models.models import Generator, BENet
from libs.models.fix_weight_dict import fix_model_state_dict
# from libs.models.cam import  GradCAM
from pytorch_grad_cam import GradCAM
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

demodata_path = './demodata'
demodata = os.listdir(demodata_path)

results_path = './results'
# result_path = './shadow_removal_image.jpg'
# image = cv2.imread("./IMG_4750.JPG")   #  "./rect_5.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# h, w, c = image.shape
m_h = 1024
m_w = 768
test_transform = Compose([Resize(m_h, m_w), Normalize(mean=(0.5, 0.5,0.5), std=(0.5, 0.5,0.5)), ToTensorV2()])

device = 'cpu'  # get_device(allow_only_gpu=False)

# benet = get_model('cam_benet', in_channels=3, pretrained=True)
model = BENet()
state_dict = torch.load("./pretrained/pretrained_benet.prm", map_location=torch.device(device))  # map_location
model.load_state_dict(fix_model_state_dict(state_dict))
model.eval()
# use WrapedGradCAM
# target_layer = model.features[3]
# benet = GradCAM(model, target_layer)

# use GradCAM
benet = model
target_layers = [benet.features[3]]
grad_cam = GradCAM(model=benet, target_layers=target_layers,
                   use_cuda=(True if device == "cuda" else False))  # alex use_cuda=False)

# srnet = get_model('srnet', pretrained=True)
# generator, discriminator = srnet[0].to(torch.device(device)), srnet[1].to(torch.device(device))
generator = Generator()
state_dict = torch.load("./pretrained/pretrained_g_srnet.prm", map_location=torch.device(device))
generator.load_state_dict(fix_model_state_dict(state_dict))

generator.eval()
generator.to(device)

figure = plt.figure(figsize=(9 * 3, 2 * 3 * len(demodata)))
dataSize = len(demodata)
for i, d in enumerate(demodata):
    im_path = os.path.join(demodata_path, d)
    print(im_path)

    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    tensor = test_transform(image=image)
    tensor = tensor['image'].unsqueeze(0).to(device)

    # color, attmap, _ = benet(tensor)
    color = benet(tensor)
    attmap = torch.from_numpy(grad_cam(tensor)).unsqueeze(dim=0)

    attmap = (attmap-0.5)/0.5

    back_color = torch.repeat_interleave(color.detach(), m_h*m_w, dim=0)
    back_ground = back_color.reshape(1, c, m_h, m_w).to(device)

    with torch.no_grad():
        input = torch.cat([tensor, attmap, back_ground], dim=1)

        tensor = tensor.detach().cpu()
        attmap = attmap.detach().cpu()
        back_ground = back_ground.detach().cpu()
        # shadow_removal_image = generator(input).detach().cpu()
        shadow_removal_image = convert_show_image(generator(input).detach().cpu().clone().numpy())

    cv2.imwrite(os.path.join(results_path, 'r_' + d), cv2.cvtColor(shadow_removal_image, cv2.COLOR_RGB2BGR))

    plt.subplot(dataSize, 4, i * 4 + 1)
    plt.title(d + ' input image')
    plt.imshow(convert_show_image(tensor.numpy()))

    plt.subplot(dataSize, 4, i * 4 + 2)
    plt.title(d + ' shadow removal image')
    plt.imshow(shadow_removal_image)

    plt.subplot(dataSize, 4, i * 4 + 3)
    plt.title(d + ' back ground color image')
    plt.imshow(convert_show_image(back_ground.numpy()))

    plt.subplot(dataSize, 4, i * 4 + 4)
    plt.title(d + ' attention map')
    plt.imshow(convert_show_image(attmap.clone().detach().cpu().numpy()), cmap='jet', alpha=0.5)
    plt.colorbar()

# plt.show()
plt.savefig('./my_plot.png')
