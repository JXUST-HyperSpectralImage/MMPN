import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import torch.nn.functional as F
     

class MMPN(nn.Module):
    def __init__(self, n_classes, in_channels,
                 H, W, patch_bands,
                 ratio=3,
                 conv_layer_k=None):
        super(MMPN, self).__init__()
        self.n_classes = n_classes
        self.C = in_channels

        self.H, self.W, self.b = H, W, patch_bands

        self.c_parts = self.C // self.b

        # TODO: use padding if not divisible. With padding, removing the BN after AvgPool will be a bit tricky. My suggestion is to keep it.
        assert self.C % self.b == 0
        self.target_shape = (-1, self.C, self.H, self.W)

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((self.C, 1, 1)),
            nn.BatchNorm3d(1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.C, self.C//ratio),
            nn.BatchNorm1d(self.C//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(self.C//ratio, self.C),
            nn.BatchNorm1d(self.C),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.b, self.b//ratio),
            nn.BatchNorm1d(self.b//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(self.b//ratio, self.b),
            nn.BatchNorm1d(self.b),
            nn.Sigmoid()
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(self.c_parts, self.c_parts, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=1),
            nn.BatchNorm3d(self.c_parts),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.c_parts, self.c_parts, kernel_size=(7, 1, 1), padding=(3, 0, 0), groups=1),
            nn.BatchNorm3d(self.c_parts),
            nn.ReLU(inplace=True)
            )

        self.fc3 = nn.Sequential(
            nn.Conv2d(self.H * self.W * self.b, self.b * self.H * self.W, 1, 1, 0, bias=True,
                      groups=self.H * self.W * self.b),
            nn.ReLU()
        )

        self.conv_layer_k = conv_layer_k

        self.conv_branch = nn.Sequential(
            nn.BatchNorm2d(self.b),
            nn.Conv2d(self.b, self.b, kernel_size=3, padding=1, bias=False, groups=self.b),
            nn.Conv2d(self.b, self.b, kernel_size=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(self.b),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.b, self.b, kernel_size=5, padding=2, bias=False, groups=self.b),
            nn.Conv2d(self.b, self.b, kernel_size=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(self.b),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.b, self.b, kernel_size=7, padding=3, bias=False, groups=self.b),
            nn.Conv2d(self.b, self.b, kernel_size=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(self.b)
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(self.b, self.b, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.b),
            nn.ReLU(inplace=True)
        )

        self.end_pool = nn.AdaptiveAvgPool3d((self.C, 1, 1))

        for f in range(self.c_parts):
            fc_classifier = nn.Sequential()
            fc_classifier.add_module('fc', nn.Linear(self.b, self.n_classes))
            # fc_classifier.add_module('relu', nn.ReLU())
            self.__setattr__('classifier{}'.format(f), fc_classifier)

    def forward(self, inputs):
        v = self.avg_pool(inputs)

        v1 = self.fc1(v.view(-1, self.C))
        v2 = self.fc2(v.view(-1, self.b))

        v = v.view(-1, 1, self.C, 1, 1) + v1.view(-1, 1, self.C, 1, 1) + v2.view(-1, 1, self.C, 1, 1)
        inputs = inputs * v
        inputs = inputs.view(-1, self.c_parts, self.b, self.H, self.W)

        partitions = self.conv3d(inputs)
        #   Feed partition map into Partition Perceptron
        fc3_inputs = partitions.reshape(-1, self.b * self.H * self.W, 1, 1)
        fc3_out = self.fc3(fc3_inputs)
        fc3_out = fc3_out.reshape(-1, self.b, self.H, self.W)

        conv_inputs = inputs.reshape(-1, self.b, self.H, self.W) + fc3_out
        
        conv_out = self.conv_branch(conv_inputs)
        conv_out = F.relu(conv_out+conv_inputs)
        conv_out = self.conv_final(conv_out)
        fc3_out += conv_out
        outputs = fc3_out.reshape(*self.target_shape)
        outputs = self.end_pool(outputs)
        
        outputs = outputs.view(-1, self.c_parts, self.b)
        pred = 0

        for c in range(self.c_parts):
            classifier = self.__getattr__('classifier{}'.format(c))
            pred += classifier(outputs[:, c])
        return pred/self.c_parts


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
    if kwargs['weights'] is None:
        weights = torch.ones(kwargs['n_classes'])
        weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
        weights = weights.to(kwargs['device'])
        kwargs.setdefault('weights', weights)
    kwargs.setdefault('patch_size', 9)

    if name == 'IndianPines':
        kwargs.setdefault('validation_percentage', 0.1)
    elif name == 'PaviaU':
        kwargs.setdefault('validation_percentage', 0.03)
    elif name == 'KSC':
        kwargs.setdefault('validation_percentage', 0.05)

    C = kwargs['n_bands']
    H = kwargs['patch_size']
    W = kwargs['patch_size']
    patch_bands = kwargs['patch_bands']
    n_classes = kwargs['n_classes']
    model = MMPN(n_classes=n_classes, in_channels=C, H=H, W=W, patch_bands=patch_bands, conv_layer_k=(3, 5, 7))
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(kwargs['device'])
    # optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001)
    # optimizer = optim.RMSprop(model.parameters(), lr=kwargs['lr'])
    optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001, momentum=0.9)

    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                        patience=kwargs['epoch'] // 10, verbose=True))
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs


if __name__ == '__main__':
    N = 1
    C = 100
    patch_bands = 20
    H = 9
    W = 9
    # h = 7
    # w = 7
    # O = 8
    groups = patch_bands * H * W
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(10, N, C, H, W).to(device)
    print('input shape is ', x.size())
    net = MMPN(10, in_channels=C, H=H, W=W, patch_bands=patch_bands, conv_layer_k=(3, 5, 7)).to(device)
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
        summary(net, (1, 100, 9, 9))
    out = net(x)
    print(out)

