import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def imshow(img):
    '''
    img: torch.Tensor
    img: [C, H, W]
    '''
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def check_model(model, num_classes=100, batch_size=4, device='cpu'):
    """
    扩展模型测试功能
    Args:
        model: 要测试的模型
        num_classes: 类别数，CIFAR-100为100
    """
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"num_classes: {num_classes}")

    # 1. 测试前向传播
    test_input = torch.randn(batch_size, 3, 32, 32).to(device)
    output = model(test_input)
    print(f"Forward pass output shape: {output.shape}")  # 应该是 [1, 100]
    
    # 2. 测试损失计算
    criterion = nn.CrossEntropyLoss()
    test_target = torch.randint(0, num_classes, (batch_size,)).to(device)     # 随机生成一个标签
    print(f"test_target shape: {test_target.shape}")
    loss = criterion(output, test_target)
    print(f"Test loss: {loss.item():.4f}")
    print("损失计算成功")
    
    # # 3. 统计模型参数
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")
    
    # # 4. 检查模型结构
    # print("\nModel architecture:")
    # print(model)
    
    # # 5. 检查每层输出形状（可选）
    # print("\nLayer output shapes:")
    # x = test_input
    # for name, layer in model.named_children():
    #     x = layer(x)
    #     print(f"{name}: {x.shape}")

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    prog_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in prog_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        top1_correct += (predicted == labels).sum().item()
        top5_correct += torch.topk(outputs, 5, dim=1).indices.eq(labels.view(-1, 1)).sum().item()
        prog_bar.set_postfix({
            'loss': running_loss / (prog_bar.n + 1),
            'top 1': top1_correct / total,
            'top 5': top5_correct / total
        })
    return running_loss / len(train_loader), top1_correct / total, top5_correct / total


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    prog_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():  # 关闭梯度计算
        for images, labels in prog_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            top5_correct += torch.topk(outputs, 5, dim=1).indices.eq(labels.view(-1, 1)).sum().item()
            prog_bar.set_postfix({
                'loss': running_loss / (prog_bar.n + 1),
                'top 1': top1_correct / total,
                'top 5': top5_correct / total
            })
    return running_loss / len(val_loader), top1_correct / total, top5_correct / total

def test_one_epoch(model, test_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    prog_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():  # 关闭梯度计算
        for images, labels in prog_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            top5_correct += torch.topk(outputs, 5, dim=1).indices.eq(labels.view(-1, 1)).sum().item()
            prog_bar.set_postfix({
                'loss': running_loss / (prog_bar.n + 1),
                'top 1': top1_correct / total,
                'top 5': top5_correct / total
            })
    return running_loss / len(test_loader), top1_correct / total, top5_correct / total


def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_epochs):
    """
    训练模型的主要函数
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备 (cpu 或 cuda)
        num_epochs: 训练轮数
    """
    best_top1 = 0.0
    best_state_dict = None
    history = {
        'train_loss': [],
        'train_top1': [],
        'train_top5': [],
        'val_loss': [],
        'val_top1': [],
        'val_top5': [],
        'test_loss': [],
        'test_top1': [],
        'test_top5': [],
    }
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_top1, train_top5 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_top1, val_top5 = validate_one_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_top1'].append(train_top1)
        history['train_top5'].append(train_top5)
        history['val_loss'].append(val_loss)
        history['val_top1'].append(val_top1)
        history['val_top5'].append(val_top5)

        
        print(f"Train Loss: {train_loss:.4f}, Top 1: {train_top1:.4f}, Top 5: {train_top5:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Top 1: {val_top1:.4f}, Top 5: {val_top5:.4f}")

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_epoch = epoch + 1
            print(f"当前最佳top1准确率: {val_top1:.4f} in epoch {epoch+1}")
            best_state_dict = model.state_dict()
    last_state_dict = model.state_dict()
    torch.save(best_state_dict, f"best_model_top1:{best_top1:.4f}_in_epoch{best_epoch}.pth")
    torch.save(last_state_dict, f"last_model_top1:{val_top1:.4f}_in_epoch{epoch+1}.pth")
    
    # 测试模型
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"加载最佳模型，进行测试")
    else:
        print("没有找到最佳模型，使用最后一次训练的模型")

    test_loss, test_top1, test_top5 = test_one_epoch(model, test_loader, criterion, device)
    history['test_loss'].append(test_loss)
    history['test_top1'].append(test_top1)
    history['test_top5'].append(test_top5)
    print(f"Test Loss: {test_loss:.4f}, Top 1: {test_top1:.4f}, Top 5: {test_top5:.4f}")
    
    return history
