"""
This is pretty hard-coded, and only serves to simplifying load different models.
All models are loaded to take image inputs bounded by [0, 1.], instead of [0, 255].
Model outputs are usually logits, but sometimes it could be in huggingface ModelOutput.
"""

import torch

from forward import LambdaNet, forward_with_cifar10_transform, forward_with_sampling, forward_with_certified
from sampling import sample_uniform_linf_with_soft_clamp


def load_model(model_name):
    if model_name == 'mnistcnn':
        return torch.hub.load('cestwc/models', 'convnet_mnist')
    elif model_name == 'mnistcnn_mnist_dataaug':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)
    
    elif model_name == 'cnn7':
        return torch.hub.load('cestwc/models', 'cnn7')
    elif model_name == 'cnn7_mnist_shi_70':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name, in_ch=1, in_dim=28)


    elif model_name == 'resnet18_cifar10_trades_100':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'cifarresnet110_cifar10_rs012':
        return torch.hub.load('cestwc/models', 'cifarresnet110', pretrained=model_name)

    elif model_name == 'wideresnet8_cifar10_shi_70':
        return torch.hub.load('cestwc/models', 'wide_resnet_8', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_trades':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'cnn7_cifar10_shi_160':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_trades_220':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_prl':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_randsmoothing':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'convmedbig_cifar10_colt_2_2_4_250':
        return torch.hub.load('cestwc/models', 'convmedbig', pretrained=model_name, width1=2, width2=2, width3=4, linear_size=250)

    elif model_name == 'resnet18_svhn_randsmoothing':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'cnn7_cifar10_sabr2':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name)

    elif model_name == 'cifarresnet110_cifar10_rs025':
        return torch.hub.load('cestwc/models', 'cifarresnet110', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_mart':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'convmed_cifar10_colt_2_4_250':
        return torch.hub.load('cestwc/models', 'convmed', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_prl':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_svhn_trades':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'convmed_mnist_colt_2_4_250':
        return torch.hub.load('cestwc/models', 'convmed', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_var_268':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'resnet18_svhn_trades_158':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'mnistcnn_trades':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_trades':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'convmed_mnist_colt_2_2_100':
        return torch.hub.load('cestwc/models', 'convmed', pretrained=model_name, width2=2, linear_size=100)

    elif model_name == 'mnistcnn_mnist_var_268_2':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'resnet18_svhn_trades_005':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'cnn7_cifar10_shi_70':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_var_035':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_randsmoothing':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_rand_669':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_svhn_erm':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_dataaug':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'mnistcnn_randsmoothing':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'wideresnet8_cifar10_shi_160':
        return torch.hub.load('cestwc/models', 'wide_resnet_8', pretrained=model_name, num_classes = 10)

    elif model_name == 'resnet18_cifar10_var_1000':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_svhn_dataaug':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_var_269':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_erm9447':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'cnn7_cifar10_sabr8':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_erm':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_var_126':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'cnn7_mnist_sabr01':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name, in_ch=1, in_dim=28)

    elif model_name == 'resnet18_cifar10_var':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_svhn_var':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'mnistcnn_mnist_trades_085':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'cnn7_mnist_sabr03':
        return torch.hub.load('cestwc/models', 'cnn7', pretrained=model_name, in_ch=1, in_dim=28)

    elif model_name == 'mnistcnn_mnist_var':
        return torch.hub.load('cestwc/models', 'mnistcnn', pretrained=model_name)

    elif model_name == 'resnet18_cifar10_erm':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_svhn_erm_2':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)

    elif model_name == 'resnet18_svhn_prl115':
        return torch.hub.load('cestwc/models', 'resnet18', pretrained=model_name)



    elif model_name.startswith('sample_4_linf_'):
        model = load_model(model_name.replace('sample_4_linf_', ''))
        if 'mnist' in model_name:
            return LambdaNet(model, forward_with_sampling, microbatch_size = 2048, sample_ = sample_uniform_linf_with_soft_clamp, eps = 0.3, num = 4)
        else:
            return LambdaNet(model, forward_with_sampling, microbatch_size = 2048, sample_ = sample_uniform_linf_with_soft_clamp, eps = 8/255, num = 4)

    elif model_name.startswith('certify_linf_'):
        model = load_model(model_name.replace('certify_linf_', ''))
        return LambdaNet(model, forward_with_certified, alpha = 1e-2, mu = 5e-2, pop=2048, sample_ = sample_uniform_linf_with_soft_clamp, eps = 8/255)

    else:
        assert False