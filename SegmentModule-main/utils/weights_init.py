import torch.nn as nn

def weights_init(m, init_type='normal'):
    classname = m.__class__.__name__

    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_fn = {
            'normal': nn.init.normal_,
            'xavier': nn.init.xavier_normal_,
            'kaiming': nn.init.kaiming_normal_,
            'orthogonal': nn.init.orthogonal_
        }.get(init_type)

        if init_fn is None:
            raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')

        init_fn(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif isinstance(m, nn.Linear):
        init_fn = {
            'normal': nn.init.normal_,
            'xavier': nn.init.xavier_normal_,
            'kaiming': nn.init.kaiming_normal_,
            'orthogonal': nn.init.orthogonal_
        }.get(init_type)

        if init_fn is None:
            raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')

        init_fn(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def init_model(net, init_type='kaiming'):
    """Initialize the model's weights with a specific initialization method."""
    net.apply(lambda m: weights_init(m, init_type))
