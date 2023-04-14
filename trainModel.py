import torchvision
import torch
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights



from torch.utils.data import DataLoader
from src.dataLoading import playersDataset, collate_fn   # these are custom for our dataset



import time
import os
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import json



num_classes = 3 # 1=ball, 2=player


# set the device (GPU is much faster)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)



num_workers = 4 if torch.cuda.is_available() else 0
batch_size = 64 # LOWER THIS IF NEEDED!

train_dir = "Data/train/"
valid_dir = "Data/valid/"
test_dir = "Data/valid/"
train_dataset = playersDataset(train_dir)
validation_dataset = playersDataset(valid_dir)
test_dataset = playersDataset(test_dir)

# Define the dataloaders for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



# these need changing, probably


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    
    parser.add_argument("--epochs", default=40, type=int, metavar="N", help="number of total epochs to run")
    
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--train-backbone", default=False, type=bool, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="initial learning rate, 0.02 is the default value for training",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    #parser.add_argument(
    #    "--lr-steps",
    #    default=[16, 22],
    #    nargs="+",
    #    type=int,
    #    help="decrease lr every step-size epochs (multisteplr scheduler only)",
    #)
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument("--output-dir", default="Results", type=str, help="path to save outputs")

    return parser

def train_model(model, criterion, optimizer, args, scheduler=None, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    losses_train = []
    accuracies = []
    losses_val = []
    accuracies_val = []
    classification_loss_train = []
    regression_loss_train = []
    classification_loss_val = []
    regression_loss_val = []

    
    os.makedirs(args.output_dir, exist_ok = True)
    modelsFolder = os.path.join(args.output_dir, "Models")
    jsonFolder = os.path.join(args.output_dir, "JSONs")
    os.makedirs(modelsFolder, exist_ok = True)
    os.makedirs(jsonFolder, exist_ok = True)

    model_name = f"SSDLite_:{args.opt.lower()}_epochs:{args.epochs}_train_backbone:{args.train_backbone}_lr:{args.lr}_lrstepsize:{args.lr_step_size}_lrgamma:{args.lr_gamma}_mom:{args.momentum}"

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            datasetSize = None
            if phase == 'train':
                model.train()  # Set model to training mode
                dataSource = train_loader
                dataset_size = len(train_dataset)
                print("TRAIN")
            else:
                model.train()
                # model.eval()   # Set model to evaluate mode
                dataSource = val_loader
                dataset_size = len(validation_dataset)
                print("EVAL")

            running_loss = 0.0
            running_corrects = 0
            classification_loss = 0.0
            regression_loss = 0.0

            # Iterate over data.
            for i, (images, targets) in enumerate(tqdm(dataSource)):

                # zero the parameter gradients
                optimizer.zero_grad()

                # send both the input images and output targets to the device
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss_dict = {}
                    if(phase == 'train'): 
                        loss_dict = model(images, targets)
                    elif(phase == 'val'):
                        with torch.no_grad():
                            loss_dict = model(images, targets)
                    # print(loss_dict)
                    losses = sum(loss for loss in loss_dict.values())  # sum the loss for all images of this epoch

                    # print(losses)
                    running_loss += float(losses)
                    classification_loss += loss_dict['classification']
                    regression_loss += loss_dict['bbox_regression']


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        losses.backward()
                        optimizer.step()

                del images
                del targets
                torch.cuda.empty_cache()

            if(phase == 'train' and scheduler != None):
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            if(phase == 'train'):
                losses_train.append(epoch_loss)
                classification_loss_train.append(classification_loss)
                regression_loss_train.append(regression_loss)
    
                
                # accuracies.append(epoch_acc)
            elif(phase == 'val'):
                losses_val.append(epoch_loss)
                classification_loss_val.append(classification_loss)
                regression_loss_val.append(regression_loss)
                # accuracies_val.append(epoch_acc)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            """
            if phase == 'val' and epoch_loss < best_loss:
                print(f'Best loss value epoch {epoch}')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint = {
                    "model": best_model_wts,
                    "optimizer": optimizer.state_dict(),
                    "args": args,
                    "epoch": epoch,
                }
                torch.save(checkpoint, os.path.join(modelsFolder, model_name+f"_{epoch}.pth"))
            """
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best loss Acc: {best_loss:4f}')
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "args": args,
        "epoch": epoch,
        "losses_train": losses_train,
        "accuracies_train": accuracies,
        "losses_val": losses_val,
        "accuracies_val": accuracies_val,
        "classification_loss_train": classification_loss_train,
        "regression_loss_train": regression_loss_train,
        "classification_loss_val": classification_loss_val,
        "regression_loss_val": regression_loss_val
    }
    torch.save(checkpoint, os.path.join(modelsFolder, model_name+"_final.pth"))

    # save the results file as a JSON
    results_dict = {"losses_train":losses_train, "accuracies":accuracies, "losses_val":losses_val, "accuracies_val":accuracies_val}

    with open(os.path.join(jsonFolder, model_name+f"_results.json"), "w") as jsonFile:
        json.dump(results_dict, jsonFile, sort_keys=True, indent=4) 



    return model, losses_train, accuracies, losses_val, accuracies_val


def createModel(args):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT)
    
    # freeze all layers, except the output heads    
    if args.train_backbone == False:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.roi_heads.parameters():
            param.requires_grad = True 
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    torch.cuda.empty_cache()
    
    model = createModel(args)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    print(model)
    
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and Adam are supported.")
        
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    num_epochs = args.epochs
    model, losses, accuracies, losses_val, accuracies_val = train_model(model, criterion, optimizer, args, scheduler, num_epochs=num_epochs)
    torch.cuda.empty_cache()