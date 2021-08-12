import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import optim
from torchsummary import summary


class MMPN(nn.Module):
    def __init__(self, n_classes, in_channels,
                 H, W, patch_bands,
                 conv_layer_k=None,
                 fc1_fc2_reduction=1,
                 fc3_groups=1):
        super().__init__()
        self.n_classes = n_classes
        self.C = in_channels
        # self.O = out_channels
        self.fc3_groups = fc3_groups

        self.H, self.W, self.b = H, W, patch_bands

        self.c_parts = self.C // self.b

        # TODO: use padding if not divisible. With padding, removing the BN after AvgPool will be a bit tricky. My suggestion is to keep it.
        assert self.C % self.b == 0
        self.target_shape = (-1, self.C, self.H, self.W)

        self.need_global_perceptron = (self.C != patch_bands)
        print('need_global_perceptron==>', self.need_global_perceptron)
        if self.need_global_perceptron:
            internal_neurons = int(self.H * self.W * self.c_parts // fc1_fc2_reduction)
            
            self.fc1_fc2 = nn.Sequential()
#            self.fc1_fc2.add_module('fc1', nn.Linear(self.H * self.W * 1, internal_neurons))
            self.fc1_fc2.add_module('fc1', nn.Linear(self.H * self.W * self.c_parts, internal_neurons))
            self.fc1_fc2.add_module('relu', nn.GELU())
#            self.fc1_fc2.add_module('fc2', nn.Linear(internal_neurons, self.H * self.W * 1))
            self.fc1_fc2.add_module('fc2', nn.Linear(internal_neurons, self.H * self.W * self.c_parts))
            
            self.avg = nn.Sequential()
            self.avg.add_module('bn', nn.BatchNorm2d(num_features=self.b))
            self.avg.add_module('avg', nn.AdaptiveAvgPool3d((1, self.H, self.W)))
#            self.avg.add_module('avg', nn.AdaptiveAvgPool2d((self.H, self.W)))
#            self.avg = nn.AdaptiveAvgPool3d((1, self.H, self.W))
#            self.avg_bn = nn.BatchNorm3d(num_features=self.c_parts)

        self.fc3 = nn.Conv2d(self.H * self.W * self.b, self.b * self.H * self.W, 1, 1, 0, bias=False, groups=self.H * self.W * self.b)
        self.fc3_bn = nn.BatchNorm1d(self.b * self.H * self.W)
        self.gelu = nn.GELU()

        self.conv_layer_k = conv_layer_k

        for k in conv_layer_k:
            conv_branch = nn.Sequential()
            conv_branch.add_module('conv',
                                   nn.Conv2d(in_channels=self.b, out_channels=self.b, kernel_size=k, padding=k // 2,
                                             bias=False, groups=self.b))
            conv_branch.add_module('bn', nn.BatchNorm2d(self.b))
            self.__setattr__('conv_bn{}'.format(k), conv_branch)
        self.end_pool = nn.AdaptiveAvgPool3d((self.C, 1, 1))
        self.end_fc = nn.Linear(self.C, self.n_classes)
        
    def forward(self, inputs):
        if self.need_global_perceptron:
            v = inputs.reshape(-1, self.b, self.H, self.H)
            v = self.avg(v)
#            v = v.reshape(-1, 1 * self.H * self.W)
            v = v.reshape(-1, self.c_parts * self.H * self.W)
            v = self.fc1_fc2(v)
            v = v.reshape(-1, self.c_parts, 1, self.H, self.W)
            v = v.repeat(1, 1, self.b, 1, 1)
            inputs = inputs.reshape(-1, self.c_parts, self.b, self.H, self.W)
#            inputs = inputs.mul(torch.sigmoid(v))
            inputs = inputs + v
        else:
            inputs = inputs.reshape(self.c_parts, self.b, self.H, self.W)

#        partitions = inputs.permute(0, 2, 4, 1, 3, 5)  # N, h_parts, w_parts, C, in_h, in_w
        partitions = inputs
        #   Feed partition map into Partition Perceptron
        fc3_inputs = partitions.reshape(-1, self.b * self.H * self.W, 1, 1)
        fc3_out = self.fc3(fc3_inputs)
        fc3_out = fc3_out.reshape(-1, self.b * self.H * self.W)
        fc3_out = self.fc3_bn(fc3_out)
        fc3_out = self.gelu(fc3_out)
        fc3_out = fc3_out.reshape(-1, self.b, self.H, self.W)

        #   Feed partition map into Local Perceptron
        if self.conv_layer_k is not None:
            conv_inputs = partitions.reshape(-1, self.b, self.H, self.W)
            conv_out = 0
            for k in self.conv_layer_k:
                conv_branch = self.__getattr__('conv_bn{}'.format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, self.b, self.H, self.W)
            fc3_out += conv_out
        out = fc3_out.reshape(*self.target_shape)
        out = self.end_pool(out)
        out = out.view(out.size(0), -1)
        out = self.end_fc(out)
        return out


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters
    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    kwargs.setdefault('device', torch.device('cpu'))
    weights = torch.ones(kwargs['n_classes'])
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(kwargs['device'])
    kwargs.setdefault('weights', weights)
    kwargs.setdefault('patch_size', 9)

    if name == 'IndianPines':
        kwargs.setdefault('patch_bands', 40)
#        kwargs.setdefault('pca_bands', 200)
        kwargs.setdefault('epoch', 100)
        # training percentage and validation percentage
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.01)
    elif name == 'PaviaU':
        # bands 103
        kwargs.setdefault('patch_bands', 20)
#        kwargs.setdefault('pca_bands', 100)
        kwargs.setdefault('epoch', 100)
        # training percentage and validation percentage
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.01)
    elif name == 'KSC':
        # bands 163
        kwargs.setdefault('patch_bands', 40)
#        kwargs.setdefault('pca_bands', 160)
        kwargs.setdefault('epoch', 100)
        # training percentage and validation percentage
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.01)
    elif name == 'Botswana':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'HoustonU':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 24)
    elif name == 'Salinas':
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.05)
        # learning rate
        kwargs.setdefault('lr', 0.0001)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 8)

    C = kwargs['n_bands']
    H = kwargs['patch_size']
    W = kwargs['patch_size']
    patch_bands = kwargs['patch_bands']
    n_classes = kwargs['n_classes']
    groups = patch_bands
    model = MMPN(n_classes=n_classes, in_channels=C, H=H, W=W, patch_bands=patch_bands, conv_layer_k=(1, 3, 5), fc1_fc2_reduction=1, fc3_groups=groups)
#    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    criterion = nn.CrossEntropyLoss()
    model = model.to(kwargs['device'])
#    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    #    optimizer = optim.RMSprop(model.parameters(), lr=kwargs['lr'])
    optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001, momentum=0.9)

    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                        patience=kwargs['epoch'] // 10, verbose=True))
    #    kwargs.setdefault(
    #        'scheduler',
    #        optim.lr_scheduler.StepLR(
    #            optimizer,
    #            step_size=33333,
    #            gamma=0.1, verbose=True))
    kwargs.setdefault('supervision', 'full')
    # 使用中心像素点作为监督信息
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs


if __name__ == '__main__':
    N = 1
    C = 100
    patch_bands = 50
    H = 9
    W = 9
    # h = 7
    # w = 7
    # O = 8
    groups = patch_bands * H * W
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(N, C, H, W).to(device)
    print('input shape is ', x.size())
    net = MMPN(10, in_channels=C, H=H, W=W, patch_bands=patch_bands, conv_layer_k=(1, 3, 5), fc1_fc2_reduction=1, fc3_groups=groups).to(device)
    net.eval()
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: {}==>{:.2f}M".format(total, total / 1e6))
    with torch.no_grad():
        summary(net, x)
    out = net(x)
    print(out)

