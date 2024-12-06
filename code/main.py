import torch
import pandas as pd
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from utils import scaler  # 从 utils 导入 scaler
from dataset import FFDIDataset
from model import create_model
from train_validate import train, validate, predict

def main():
    # 判断是否可以使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_label = pd.read_csv('train.txt')
    val_label = pd.read_csv('val.txt')

    train_label['path'] = 'trainset/' + train_label['img_name']
    val_label['path'] = 'valset/' + val_label['img_name']

    model = create_model().to(device)  # 将模型转移到相应设备

    # 使用 max_samples 限制加载的数据量
    train_dataset = FFDIDataset(
        img_path=train_label['path'],
        img_label=train_label['target'],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        max_samples=100  # 假设限制为 10000 条数据
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )

    val_dataset = FFDIDataset(
        img_path=val_label['path'],
        img_label=val_label['target'],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        max_samples=1000  # 假设限制验证集为 1000 条数据
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=50)

    best_acc = 0.0
    early_stop_patience = 5
    early_stop_counter = 0

    for epoch in range(1):
        print('Epoch: ', epoch)
        train(train_loader, model, criterion, optimizer, epoch, device)
        val_loss, val_acc = validate(val_loader, model, criterion, device)

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = round(val_acc.item(), 2)
            torch.save(model.state_dict(), f'./model_{best_acc}.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    test_loader = torch.utils.data.DataLoader(
        FFDIDataset(
            img_path=val_label['path'],
            img_label=val_label['target'],
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            max_samples=1000  # 假设限制测试集为 1000 条数据
        ),
        batch_size=64, shuffle=False, num_workers=4
    )

    val_label['y_pred'] = predict(test_loader, model, 1, device)[:, 1]
    val_label[['img_name', 'y_pred']].to_csv('submit.csv', index=None)

if __name__ == '__main__':
    main()
