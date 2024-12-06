import numpy as np
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm  # 使用 tqdm 而不是 tqdm_notebook

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)  # 将数据迁移到设备
            target = target.to(device).long()  # 将数据迁移到设备并转换为 Long 类型

            output = model(input)
            loss = criterion(output, target)

            acc = (output.argmax(1).view(-1) == target.view(-1)).float().mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        return losses.avg, top1.avg

def predict(test_loader, model, tta=10, device=None):
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):  # 使用 tqdm 而不是 tqdm_notebook
                input = input.to(device)  # 将数据迁移到设备
                target = target.to(device).long()  # 将数据迁移到设备并转换为 Long 类型

                output = model(input)
                output = F.softmax(output, dim=1)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)  # 将数据迁移到设备
        target = target.to(device).long()  # 将数据迁移到设备并转换为 Long 类型

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        acc = (output.argmax(1).view(-1) == target.view(-1)).float().mean() * 100
        top1.update(acc, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)
