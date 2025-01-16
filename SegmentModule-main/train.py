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
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_soma': val_dice.avg[1]})
    return val_log


def train_one_epoch(model, train_loader, optimizer, loss_func, n_labels, alpha, device):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        loss3 = loss_func(output[3], target)

        loss = loss3 + alpha * (loss0 + loss1 + loss2)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output[3], target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_soma': train_dice.avg[1]})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('../experiments/model/', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    dataset = TrainDataset(args)
    img, seg = dataset[0]
    train_loader = DataLoader(dataset=TrainDataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=ValDataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # model info
    model = SSTNet(_conv_repr=True, _pe_type="learned").to(device)
    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    loss = loss.DiceLoss()
    # loss = loss.TverskyLoss()
    # criterion = getattr(criterions, args.criterion)

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]
    trigger = 0
    alpha = args.alpha
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train_one_epoch(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = validate(model, val_loader, loss, args.n_labels)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_soma'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_soma']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        if epoch % 30 == 0:
            alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
