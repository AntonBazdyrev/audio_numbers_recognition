import torch
import torchvision


class ClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, num_digits=6):
        super(ClassificationHead, self).__init__()
        self.heads = torch.nn.ModuleList([torch.nn.Linear(input_dim, 10) for i in range(num_digits)])
        
    def forward(self, x):
        return torch.stack([h(x) for h in self.heads]).permute(1, 2, 0)
    
def create_model():
    model = torchvision.models.shufflenet_v2_x0_5()
    conv1 = model.conv1[0]
    model.conv1[0] = torch.nn.Conv2d(
        1, conv1.out_channels, 
        kernel_size=conv1.kernel_size[0], 
        stride=conv1.stride[0], 
        padding=conv1.padding[0]
    )
    num_ftrs = model.fc.in_features
    model.fc = ClassificationHead(num_ftrs)
    return model
