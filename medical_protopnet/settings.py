base_architecture = 'vgg19'
img_size = 224 # the size to resize the image to (images in their raw form are )
prototype_shape = (50, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

mean = (2.1061, 2.1532, 2.1436) # if unknown, set to None and it will automatically be computed
std = (0.9813, 1.0032, 0.9987) # if unknown, set to None and it will automatically be computed

experiment_run = '001'

class_specific = True

data_path = '../../data/chest_xray/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

loss_coefficients = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 1

push_start = 1
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

target_accu = 0.70 # log the model if accuracy above this value