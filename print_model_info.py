import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import re
import os
from torchsummary import summary

# local imports
import model
from preprocess import mean, std, preprocess_input_function
from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from receptive_field import compute_proto_layer_rf_info_v2

if __name__ == "__main__":

    # book keeping namings and code
    from settings import base_architecture, img_size, prototype_shape, num_classes, \
                        prototype_activation_function, add_on_layers_type, experiment_run

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    # load the data
    from settings import train_dir, test_dir, train_push_dir, \
                        train_batch_size, test_batch_size, train_push_batch_size

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    if prototype_activation_function == 'linear':
       ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = ppnet.to(device)

    layer_filter_sizes, layer_strides, layer_paddings = model.features.conv_info()
    n_out, j_out, r_out, start_out = model.proto_layer_rf_info # output size, receptive field jump of output layer, receptive field size of output layer, center of receptive field of output layer

    print("ProtoPNet summary")
    summary(model, (3, 224, 224))
    print("base_architecture", base_architecture)
    print("prototype_shape", model.prototype_shape)
    print("num_prototypes", model.num_prototypes)
    print("prototype_vectors", model.prototype_vectors.shape)
    print("add_on_layers", model.add_on_layers)
    print("img_size", model.img_size)
    print("num_classes", model.num_classes)
    print("prototype_activation_function", model.prototype_activation_function)
    print("features", model.features)
    print("prototype_shape", model.prototype_shape)
    print("prototype_class_identity", model.prototype_class_identity.size())
    print("last_layer.weight.data", model.last_layer.weight.data.size())
    print("prototype_distances", model.prototype_distances)