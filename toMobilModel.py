import torch
from libs.models.models import Generator, BENet
from libs.models.fix_weight_dict import fix_model_state_dict
from torch.utils.mobile_optimizer import optimize_for_mobile
from pytorch_grad_cam import GradCAM

m_h = 1024
m_w = 768
device = 'cpu'  # get_device(allow_only_gpu=False)

benet = BENet()
state_dict = torch.load("./pretrained/pretrained_benet.prm", map_location=torch.device(device))  # map_location
benet.load_state_dict(fix_model_state_dict(state_dict))
benet.eval()

target_layers = [benet.features[3]]
grad_cam = GradCAM(model=benet, target_layers=target_layers,
                   use_cuda=(True if device == "cuda" else False))  # alex use_cuda=False)


generator = Generator()
state_dict = torch.load("./pretrained/pretrained_g_srnet.prm", map_location=torch.device(device))
generator.load_state_dict(fix_model_state_dict(state_dict))

generator.eval()
generator.to(device)

# prepare example tensor
benet_input_example_tensor = torch.rand(3, m_h, m_w).unsqueeze(0).to(device)
# generate benet model for mobile
benet_traced_module = torch.jit.trace(benet, benet_input_example_tensor)
benet_optimized_model = optimize_for_mobile(benet_traced_module)
benet_optimized_model.save('benet_model_m.pt')

# prepare example tensor of generator
color = benet(benet_input_example_tensor)
attmap = torch.from_numpy(grad_cam(benet_input_example_tensor)).unsqueeze(dim=0)
attmap = (attmap-0.5)/0.5
back_color = torch.repeat_interleave(color.detach(), m_h*m_w, dim=0)
back_ground = back_color.reshape(1, 3, m_h, m_w).to(device)
generator_input_example_tensor = torch.cat([benet_input_example_tensor, attmap, back_ground], dim=1)
# generate generator model for mobile
generator_traced_module = torch.jit.trace(generator, generator_input_example_tensor)
generator_optimized_model = optimize_for_mobile(generator_traced_module)
generator_optimized_model.save('shadow_remove_model_m.pt')
