from .resnet import ResnetBuilder

__factory = {
    'resnet18': "build_resnet_18",
    'resnet34': "build_resnet_34",
    'resnet50': 'build_resnet_50',
    'resnet101': 'build_resnet_101',
    'resnet152': 'build_resnet_152',
}

builder = ResnetBuilder()

def create_model(arch,**kwargs):
    assert arch in __factory, '{} not supported yet'.format(arch)
    network = __factory[arch]
    return getattr(builder, network)(**kwargs)


