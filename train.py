import os
import json
import time
import torch
import copy
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn

from mamvit import MamViT
from dataloader import load_data
from utils import colorstr

import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls") 
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from datetime import datetime
import torch.nn.functional as F
device = "cuda" #if torch.cuda.is_available() else 'cpu'

        

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_model(args, model, optimizer, device, num_epochs, dataloaders, path, ):
    since = time.time()
    
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    history['best_epoch'] = 0

    model.to(device)

    train_datetime = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    cur_path = os.path.join(path, train_datetime)
    os.makedirs(cur_path, exist_ok=True)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'Epoch {epoch}/{num_epochs-1}:'))

        for phase in ["train", "val"]:
            if phase == "train":
                adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                                    ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('green', 'bold', '\n%20s' + '%15s' * 3) %
                                    ('Validation:', 'gpu_mem', 'loss', 'acc'))
                model.eval()
            running_items = 0
            running_loss = 0.0
            running_corrects = 0

            _phase = tqdm(dataloaders[phase],
                      total=len(dataloaders[phase]),
                      bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                      unit='batch')

            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_items += inputs.size(0)
                try:
                    running_loss += loss.item() * inputs.size(0)
                except:
                    print(loss.item().shape, inputs.size(0))
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items

               
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss / running_items, running_corrects / running_items)
                _phase.set_description_str(desc)

            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
                
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_optim = copy.deepcopy(optimizer.state_dict())
                    history['best_epoch'] = epoch
                    # Save the best model
                    torch.save(model.state_dict(), f"{cur_path}/best.pth")
                    
                print(f"Best val {best_val_acc} at epoch {history['best_epoch']}")
            
            if phase == "train":
                # Save model for each epoch (overwrite the last model)
                torch.save(model.state_dict(), f"{cur_path}/last.pth")
                print("[INFO] Last model saved")
                
            
        
        
       
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs'.format(time_elapsed // 3600,time_elapsed % 3600 // 60, time_elapsed % 60, num_epochs))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)
    history['INFO'] = 'Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs -Best Epoch: {} - Best val Acc: {:4f}'.format(time_elapsed // 3600,time_elapsed % 3600 // 60, time_elapsed % 60, num_epochs, history['best_epoch'], best_val_acc)
    print("[INFO] Best model saved")
    
    with open(os.path.join(cur_path, 'result.json'), "w") as outfile:
        json.dump(history, outfile)

    return model, best_val_acc.item()




def experiment(args):
    
    name = f"{args.name}_{args.dataset}_{args.model}_{args.lr}_{args.batch_size}_{args.epochs}"

    exp_path = os.path.join(args.save_path, name)
    os.makedirs(exp_path, exist_ok=True)
    
    

    dataloaders = load_data(args.batch_size, args.dataset)

    if args.dataset.lower() == "cifar100":
        num_classes = 100
        args.num_classes = num_classes
    elif args.dataset.lower() == "cifar10":
        num_classes = 10
        args.num_classes = num_classes
    elif args.dataset.lower() == "flower102":
        num_classes = 102
        args.num_classes = num_classes
    elif args.dataset.lower() == "tinyimagenet200":
        num_classes = 200
        args.num_classes = num_classes
    else:
        raise Exception(f"{args.dataset} dose not know num_classes")

    model = MamViT(args, num_classes=num_classes).to(device=device)
    
    # cal #params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params}")
    
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)

    print(args)
    with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

  
    best_mode, best_val_acc = train_model(args = args,
                                      model = model,
                                      optimizer = optimizer,
                                      device = device,
                                      num_epochs = args.epochs,
                                      dataloaders = dataloaders,
                                      path=exp_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='No_Name', type=str, help="Test case name")
    parser.add_argument("--dataset", default='CIFAR100', type=str, help="CIFAR100.")
    parser.add_argument("--model", default='resnet18', type=str, help="Name of model architecure")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--save_path", default="save_path", type=str, help="Path to save model and results")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--decay", default=0.001, type=float)
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    
    
    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--dropout_rate", default=0, type=float)
    
    parser.add_argument("--d_model", default=389, type=int)
    parser.add_argument("--d_inner", default=389, type=int)
    parser.add_argument("--d_conv", default=4, type=int)
    parser.add_argument("--dt_rank", default=48, type=int)
    parser.add_argument("--d_state", default=16, type=int)
    
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--bias",  default=False, type=bool)
    parser.add_argument("--conv_bias", default=True, type=bool)

    args = parser.parse_args()

    experiment(args)

    




