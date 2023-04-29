"""
This is pretty hard-coded, and only serves to simplifying load different models.
All models are loaded to take image inputs bounded by [0, 1.], instead of [0, 255].
Model outputs are usually logits, but sometimes it could be in huggingface ModelOutput.
"""

import torch

from .nets import ConvNet, CNN7, ResNet18, ResNet50
from .convmed import ConvMed, ConvMedBig
from .wide_resnet_bn import wide_resnet_8

from forward import LambdaNet, forward_with_cifar10_transform, forward_with_sampling, forward_with_certification
from sampling import sample_uniform_linf_with_soft_clamp


def load_model(model_name):
    if model_name == 'convnet_mnist':
        return ConvNet()

    elif model_name.startswith('convnet_mnist_'):
        model = ConvNet()
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'resnet18_cifar10' or model_name == 'resnet18_svhn':
        return ResNet18()

    elif model_name.startswith('resnet18_'):
        model = ResNet18()
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'cnn7_svhn' or model_name == 'cnn7_cifar10':
        return CNN7(3, 32)

    elif model_name == 'cnn7_mnist':
        return CNN7(1, 28)

    elif model_name.startswith('cnn7_cifar10_'):
        model = LambdaNet(CNN7(3, 32), forward_with_cifar10_transform)
        model.net.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'wide_resnet_8':
        return wide_resnet_8()

    elif model_name.startswith('wide_resnet_8_cifar10_'):
        model = LambdaNet(wide_resnet_8(), forward_with_cifar10_transform)
        model.net.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'convmed_cifar10':
        return ConvMed()

    elif model_name.startswith('convmed_cifar10_'):
        model = ConvMed()
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'convmed_mnist':
        return ConvMed(dataset='mnist', input_size=28, input_channel=1)

    elif model_name == 'convmed_mnist_colt_2_4_250':
        model = ConvMed(dataset='mnist', input_size=28, input_channel=1)
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'convmed_mnist_colt_2_2_100':
        model = ConvMed(dataset='mnist', input_size=28, input_channel=1, width1=2, width2=2, linear_size=100)
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'cifarresnet110':
        import pytorchcv.model_provider
        return pytorchcv.model_provider.get_model(f"resnet110_cifar10", pretrained=False)

    elif model_name == 'cifarresnet110_cifar10':
        import pytorchcv.model_provider
        return pytorchcv.model_provider.get_model(f"resnet110_cifar10", pretrained=True)

    elif model_name.startswith('cifarresnet110_cifar10_'):
        import pytorchcv.model_provider
        model = pytorchcv.model_provider.get_model(f"resnet110_cifar10", pretrained=False)
        model.load_state_dict(torch.load('pretrained/' + model_name + '.pt'))
        return model

    elif model_name == 'cifarwrn28_10_cifar100':
        import pytorchcv.model_provider
        return pytorchcv.model_provider.get_model(f"wrn28_10_cifar100", pretrained=False)


    elif model_name.startswith('sample_4_linf_'):
        model = load_model(model_name.replace('sample_4_linf_', ''))
        if 'mnist' in model_name:
            return LambdaNet(model, forward_with_sampling, microbatch_size = 2048, sample_ = sample_uniform_linf_with_soft_clamp, eps = 0.3, num = 4)
        else:
            return LambdaNet(model, forward_with_sampling, microbatch_size = 2048, sample_ = sample_uniform_linf_with_soft_clamp, eps = 8/255, num = 4)

    elif model_name.startswith('certify_linf_'):
        model = load_model(model_name.replace('certify_linf_', ''))
        return LambdaNet(model, forward_with_certification, alpha = 1e-2, mu = 5e-2, pop=2048, sample_ = sample_uniform_linf_with_soft_clamp, eps = 8/255)

    else:
        assert False