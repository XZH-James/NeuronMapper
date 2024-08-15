import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

import config
from models.NMSeg import SSTNet
from dataset.data_soma_train import TrainDataset
from dataset.data_soma_val import ValDataset
from utils import logger, weights_init, metrics, common, loss


def validate(model, val_loader, loss_func, n_labels, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", total=len(val_loader)):
            data, target = data.float().to(device), target.long().to(device)
            target = common.to_one_hot_3d(target, n_labels)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)

    return OrderedDict({
        'Val_Loss': val_loss.avg,
        'Val_dice_soma': val_dice.avg[1]
    })


def train_one_epoch(model, train_loader, optimizer, loss_func, n_labels, alpha, device):
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for data, target in tqdm(train_loader, desc="Training", total=len(train_loader)):
        data, target = data.float().to(device), target.long().to(device)
        target = common.to_one_hot_3d(target, n_labels)

        optimizer.zero_grad()
        outputs = model(data)
        losses = [loss_func(output, target) for output in outputs]
        total_loss = losses[-1] + alpha * sum(losses[:-1])
        total_loss.backward()
        optimizer.step()

        train_loss.update(total_loss.item(), data.size(0))
        train_dice.update(outputs[-1], target)

    return OrderedDict({
        'Train_Loss': train_loss.avg,
        'Train_dice_soma': train_dice.avg[1]
    })


def save_checkpoint(state, save_path, filename='latest_model.pth'):
    torch.save(state, os.path.join(save_path, filename))


def main():
    args = config.args
    save_path = os.path.join('../', args.save)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cpu' if args.cpu else 'cuda')

    # Data loaders
    train_loader = DataLoader(TrainDataset(args), batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(ValDataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # Model setup
    model = SSTNet(_conv_repr=True, _pe_type="learned").to(device)
    model.apply(weights_init.init_model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_func = loss.DiceLoss()
    log = logger.Train_Logger(save_path, "train_log")

    best_epoch, best_dice = 0, 0
    trigger = 0
    alpha = args.alpha

    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        print(f"======= Epoch: {epoch} | LR: {optimizer.param_groups[0]['lr']} =======")

        train_log = train_one_epoch(model, train_loader, optimizer, loss_func, args.n_labels, alpha, device)
        val_log = validate(model, val_loader, loss_func, args.n_labels, device)
        log.update(epoch, train_log, val_log)

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        save_checkpoint(state, save_path)

        if val_log['Val_dice_soma'] > best_dice:
            print('Saving best model...')
            save_checkpoint(state, save_path, 'best_model.pth')
            best_epoch, best_dice = epoch, val_log['Val_dice_soma']
            trigger = 0
        else:
            trigger += 1

        print(f'Best performance at Epoch: {best_epoch} | Dice: {best_dice}')

        if epoch % 30 == 0:
            alpha *= 0.8

        if args.early_stop is not None and trigger >= args.early_stop:
            print("=> Early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
