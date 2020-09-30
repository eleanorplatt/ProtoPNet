import os
import shutil
import argparse
import re

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader

import model
import train_and_test
import push

if __name__ == "__main__":

    # GPU set-up
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python main.py -gpuid=0,1,2,3
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print('GPU ID: ', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # create directory to save models, results and logs
    from settings import base_architecture, experiment_run
    from utils import create_logger, makedir
    model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)

    # compute mean and standard deviation of training dataset if necessary
    from settings import mean, std, train_dir, train_batch_size, img_size
    from utils import compute_mean_std
    if (mean == None) or (std == None):
        print('Computing mean and standard deviation of training data...')
        mean, std = compute_mean_std(train_dir, train_batch_size, img_size)
    
    # set-up datasets and dataloaders
    from settings import train_push_dir, train_push_batch_size, test_dir, test_batch_size
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    # log information about the datasets
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))
    
    # construct the model
    from settings import prototype_shape, num_classes, prototype_activation_function, add_on_layers_type
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet).to(device)
    
    # define optimiser
    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularised
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    from settings import loss_coefficients, num_train_epochs, num_warm_epochs, push_start, push_epochs, class_specific, \
        weight_matrix_filename, prototype_img_filename_prefix, prototype_self_act_filename_prefix, proto_bound_boxes_filename_prefix, \
            target_accu
    from utils import save_model_w_condition, preprocess_input_function
    log('start training')
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            train_and_test.warm_only(model=ppnet_multi, log=log)
            _ = train_and_test.train(model=ppnet_multi, device=device, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, loss_coefficients=loss_coefficients, log=log)
        else:
            train_and_test.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = train_and_test.train(model=ppnet_multi, device=device, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, loss_coefficients=loss_coefficients, log=log)

        accu = train_and_test.test(model=ppnet_multi, device=device, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=target_accu, log=log)

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                mean=mean,
                std=std,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = train_and_test.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=target_accu, log=log)

            if prototype_activation_function != 'linear':
                train_and_test.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = train_and_test.train(model=ppnet_multi, device=device, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, loss_coefficients=loss_coefficients, log=log)
                    accu = train_and_test.test(model=ppnet_multi, device=device, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save_model_w_condition(model=ppnet, device=device, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=target_accu, log=log)
    
    logclose()