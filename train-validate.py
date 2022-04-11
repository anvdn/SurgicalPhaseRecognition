import argparse
import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=.005)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--momentum', type=float, default=.9)  #maybe decrease
    parser.add_argument('--gamma', type=float, default=.4)
    parser.add_argument('--step_size', type=int, default=1)  
    parser.add_argument('--os', type=int, default=0)  
    args = parser.parse_args()

    # set model name
    model_name = 'MobileNet_l' + str(args.lr) + '_e' + str(args.epochs) + '_g' + str(args.gamma) + '_s' + str(args.step_size) + '_os' + str(args.os)

    # set model parameters
    model_parameters = (model_name, utils.num_classes, True, 20)

    # load model
    model = models.MobileNetStage(*model_parameters).to(utils.device)

    # create pytorch datasets
    datasets = {x: utils.HerniaDataset(utils.dfs_path + '/' + x + '_no_temp' + ((x == 'training' and args.os == 1) * '_os') + '.pkl', is_stage_feature = True, num_stages = 20,
                transform = utils.data_transforms[x]) for x in ['training', 'validation']}

    # instantiate data loaders
    dataloaders = {x: utils.DataLoader(dataset=datasets[x], batch_size=args.batch_size, shuffle=True) for x in ['training', 'validation']}

    # criterion is cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # decay LR by a factor GAMMA every STEP_SIZE epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    utils.train_model(model = model, 
                        model_name = model.model_name,  #  name of the model which will be the name of the saved weights file within /weights
                        dataloaders = dataloaders, 
                        criterion = criterion, 
                        optimizer = optimizer, 
                        scheduler = exp_lr_scheduler, 
                        num_epochs=args.epochs,
                        validation = True)