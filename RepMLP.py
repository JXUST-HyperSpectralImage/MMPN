import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import optim
from torchsummary import summary


def fuse_bn(conv_or_fc, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    if conv_or_fc.weight.ndim == 4:
        t = t.reshape(-1, 1, 1, 1)
    else:
        t = t.reshape(-1, 1)
    return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std


class RepMLP(nn.Module):
    def __init__(self, in_channels,
                 H, W, patch_bands,
                 reparam_conv_k=None,
                 fc1_fc2_reduction=1,
                 fc3_groups=1,
                 deploy=False, ):
        super().__init__()

        self.C = in_channels
        # self.O = out_channels
        self.fc3_groups = fc3_groups

        self.H, self.W, self.b = H, W, patch_bands

        self.c_parts = self.C // self.b

        # TODO: use padding if not divisible. With padding, removing the BN after AvgPool will be a bit tricky. My suggestion is to keep it.
        assert self.C % self.b == 0
        self.target_shape = (-1, self.b, self.H, self.W)

        self.deploy = deploy

        self.need_global_perceptron = (C != patch_bands)
        if self.need_global_perceptron:
            internal_neurons = int(self.H * self.W * self.c_parts // fc1_fc2_reduction)
            self.fc1_fc2 = nn.Sequential()
            self.fc1_fc2.add_module('fc1', nn.Linear(self.H * self.W * self.c_parts, internal_neurons))
            self.fc1_fc2.add_module('relu', nn.ReLU())
            self.fc1_fc2.add_module('fc2', nn.Linear(internal_neurons, self.H * self.W * self.c_parts))
            if deploy:
                self.avg = nn.AdaptiveAvgPool3d((self.c_parts, self.H, self.W))
            else:
                self.avg = nn.Sequential()
                self.avg.add_module('avg', nn.AdaptiveAvgPool3d((self.c_parts, self.H, self.W)))
                self.avg.add_module('bn', nn.BatchNorm2d(num_features=self.c_parts))

        self.fc3 = nn.Conv2d(self.H * self.W * self.b, self.b * self.H * self.W, 1, 1, 0, bias=deploy,
                             groups=fc3_groups)
        self.fc3_bn = nn.Identity() if deploy else nn.BatchNorm1d(self.b * self.H * self.W)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = nn.Sequential()
                conv_branch.add_module('conv',
                                       nn.Conv2d(in_channels=self.b, out_channels=self.b, kernel_size=k, padding=k // 2,
                                                 bias=False, groups=fc3_groups))
                conv_branch.add_module('bn', nn.BatchNorm2d(self.b))
                self.__setattr__('repconv{}'.format(k), conv_branch)

    def forward(self, inputs):
        if self.need_global_perceptron:
            v = self.avg(inputs)
            v = v.reshape(-1, self.c_parts * self.H * self.W)
            v = self.fc1_fc2(v)
            v = v.reshape(self.c_parts, -1, 1, self.H, 1, self.W)
            v = v.repeat(1, self.b, 1, 1, 1, 1)
            inputs = inputs.reshape(-1, self.b, 1, self.H, 1, self.W)
            inputs = inputs + v
        else:
            inputs = inputs.reshape(-1, self.b, 1, self.H, 1, self.W)

        partitions = inputs.permute(0, 2, 4, 1, 3, 5)  # N, h_parts, w_parts, C, in_h, in_w

        #   Feed partition map into Partition Perceptron
        fc3_inputs = partitions.reshape(-1, self.b * self.H * self.W, 1, 1)
        fc3_out = self.fc3(fc3_inputs)
        fc3_out = fc3_out.reshape(-1, self.b * self.H * self.W)
        fc3_out = self.fc3_bn(fc3_out)
        fc3_out = fc3_out.reshape(-1, 1, 1, self.b, self.H, self.W)

        #   Feed partition map into Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.b, self.H, self.W)
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)

            conv_out = conv_out.reshape(-1, 1, 1, self.b, self.H, self.W)
            fc3_out += conv_out
        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*self.target_shape)
        return out

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        I = torch.eye(self.b * self.H * self.W // self.fc3_groups).repeat(1, self.fc3_groups).reshape(
            self.b * self.H * self.W // self.fc3_groups, self.b, self.H, self.W).to(conv_kernel.device)
        fc_k = F.conv2d(I, conv_kernel, padding=conv_kernel.size(2) // 2, groups=self.fc3_groups)
        fc_k = fc_k.reshape(self.b * self.H * self.W // self.fc3_groups, self.b * self.H * self.W).t()
        fc_bias = conv_bias.repeat_interleave(self.H * self.W)
        return fc_k, fc_bias

    def get_equivalent_fc1_fc3_params(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__('repconv{}'.format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__('repconv{}'.format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        if self.need_global_perceptron:
            avgbn = self.avg.bn
            std = (avgbn.running_var + avgbn.eps).sqrt()
            scale = avgbn.weight / std
            avgbias = avgbn.bias - avgbn.running_mean * scale
            fc1 = self.fc1_fc2.fc1
            replicate_times = fc1.in_features // len(avgbias)
            replicated_avgbias = avgbias.repeat_interleave(replicate_times).view(-1, 1)
            bias_diff = fc1.weight.matmul(replicated_avgbias).squeeze()
            fc1_bias_new = fc1.bias + bias_diff
            fc1_weight_new = fc1.weight * scale.repeat_interleave(replicate_times).view(1, -1)
        else:
            fc1_bias_new = None
            fc1_weight_new = None

        return fc1_weight_new, fc1_bias_new, final_fc3_weight, final_fc3_bias

    def switch_to_deploy(self):
        self.deploy = True
        fc1_weight, fc1_bias, fc3_weight, fc3_bias = self.get_equivalent_fc1_fc3_params()
        #   Remove Local Perceptron
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__('repconv{}'.format(k))
        #   Remove the BN after FC3
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(self.b * self.H * self.W, self.b * self.H * self.W, 1, 1, 0, bias=True,
                             groups=self.fc3_groups)
        self.fc3_bn = nn.Identity()
        #   Remove the BN after AVG
        if self.need_global_perceptron:
            self.__delattr__('avg')
            self.avg = nn.AdaptiveAvgPool3d((self.c_parts, self.H, self.W))
        #   Set values
        if fc1_weight is not None:
            self.fc1_fc2.fc1.weight.data = fc1_weight
            self.fc1_fc2.fc1.bias.data = fc1_bias
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias


def repmlp_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


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
    kwargs.setdefault('patch_size', 7)

    if name == 'IndianPines':
        kwargs.setdefault('patch_bands', 50)
        kwargs.setdefault('epoch', 100)
        # training percentage and validation percentage
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('training_percentage', 0.1)
        kwargs.setdefault('validation_percentage', 0.1)
        # learning rate
        kwargs.setdefault('lr', 0.01)
    elif name == 'PaviaU':
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 64)
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.03)
        kwargs.setdefault('validation_percentage', 0.03)
        # learning rate
        kwargs.setdefault('lr', 0.01)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 1)
    elif name == 'KSC':
        kwargs.setdefault('epoch', 130)
        kwargs.setdefault('batch_size', 32)
        # training percentage and validation percentage
        kwargs.setdefault('training_percentage', 0.03)
        kwargs.setdefault('validation_percentage', 0.03)
        # learning rate
        kwargs.setdefault('lr', 0.01)
        # conv layer kernel numbers
        kwargs.setdefault('kernel_nums', 1)
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

    C = kwargs['bands']
    H = kwargs['patch_size']
    W = kwargs['patch_size']
    patch_bands = kwargs['patch_bands']
    groups = patch_bands
    model = RepMLP(C, H=H, W=W, patch_bands=patch_bands, reparam_conv_k=(1, 3, 5), fc1_fc2_reduction=1, fc3_groups=groups,
                    deploy=False)
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
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
    C = 200
    patch_bands = 50
    H = 7
    W = 7
    # h = 7
    # w = 7
    # O = 8
    groups = patch_bands
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(N, C, H, W).to(device)
    print('input shape is ', x.size())
    repmlp = RepMLP(C, H=H, W=W, patch_bands=patch_bands, reparam_conv_k=(1, 3, 5), fc1_fc2_reduction=1, fc3_groups=groups,
                    deploy=False).to(device)
    repmlp.eval()
    for module in repmlp.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)

    total = sum([param.nelement() for param in repmlp.parameters()])
    print("Number of parameter: {}==>{:.2f}M".format(total, total / 1e6))
    with torch.no_grad():
        summary(repmlp, x)
    out = repmlp(x)

    repmlp.switch_to_deploy()
    total = sum([param.nelement() for param in repmlp.parameters()])
    print("Number of parameter: {}==>{:.2f}M".format(total, total / 1e6))
    with torch.no_grad():
        summary(repmlp, x)

    deployout = repmlp(x)
    # 参数重构之后网络与原始网络误差
    print('difference between the outputs of the training-time and converted RepMLP is')
    print(((deployout - out) ** 2).sum())
