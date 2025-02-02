import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json

def imshow(img):
    '''
    img: torch.Tensor
    img: [C, H, W]
    '''
    img = img / 2 + 0.5 # 归一化到 [0, 1]
    npimg = img.numpy() # 转换为 numpy 数组
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # 调整维度顺序为 (H, W, C) 并显示
    plt.show() # 显示图像

def check_model(model, num_classes=100, batch_size=4, device='cpu'):
    """
    扩展模型测试功能
    Args:
        model: 要测试的模型
        num_classes: 类别数，CIFAR-100为100
        batch_size: 批量大小
        device: 设备 (cpu 或 cuda)
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

def train_one_epoch(model, train_loader, optimizer, criterion, device, save_log=False, save_dir=None, epoch=None, batch_index=0, log_file=None):
    """
    训练一个epoch
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备 (cpu 或 cuda)
        save_log: 是否保存日志
        save_dir: 日志保存路径
        epoch: 当前epoch数
        batch_index: 当前batch索引
        log_file: 日志文件
    """
    model.train() # 设置模型为训练模式
    running_loss = 0.0 # 初始化累计损失
    top1_correct = 0 # 初始化top1正确数量
    top5_correct = 0 # 初始化top5正确数量
    total = 0 # 初始化总样本数
    
    prog_bar = tqdm(train_loader, desc="Training", leave=False) # 创建进度条
    for images, labels in prog_bar: # 遍历训练数据加载器
        images, labels = images.to(device), labels.to(device) # 将数据加载到指定设备
        optimizer.zero_grad() # 清空梯度
        outputs = model(images) # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        running_loss += loss.item() # 累加损失
        _, predicted = torch.max(outputs.data, 1) # 获取预测结果
        total += labels.size(0) # 累加样本总数
        top1_correct += (predicted == labels).sum().item() # 累加top1正确数量
        top5_correct += torch.topk(outputs, 5, dim=1).indices.eq(labels.view(-1, 1)).sum().item() # 累加top5正确数量
        
        batch_loss = running_loss / (prog_bar.n + 1) # 计算当前batch的平均损失
        batch_top1 = top1_correct / total # 计算当前batch的top1准确率
        batch_top5 = top5_correct / total # 计算当前batch的top5准确率
        
        prog_bar.set_postfix({ # 更新进度条显示
            'loss': batch_loss,
            'top 1': batch_top1,
            'top 5': batch_top5
        })
        
        if save_log: # 如果需要保存日志
            log_data = {
                "type": "batch_train", # 日志类型
                "epoch": epoch, # 当前epoch数
                "batch": batch_index, # 当前batch索引
                "loss": batch_loss, # 当前batch的平均损失
                "top1": batch_top1, # 当前batch的top1准确率
                "top5": batch_top5, # 当前batch的top5准确率
                "timestamp": datetime.now().isoformat() # 当前时间戳
            }
            log_file.write(json.dumps(log_data) + "\n") # 将日志写入文件
        batch_index += 1 # batch索引加1
    
    epoch_loss = running_loss / len(train_loader) # 计算当前epoch的平均损失
    epoch_top1 = top1_correct / total # 计算当前epoch的top1准确率
    epoch_top5 = top5_correct / total # 计算当前epoch的top5准确率
    
    if save_log: # 如果需要保存日志
        log_data = {
            "type": "epoch_train", # 日志类型
            "epoch": epoch, # 当前epoch数
            "loss": epoch_loss, # 当前epoch的平均损失
            "top1": epoch_top1, # 当前epoch的top1准确率
            "top5": epoch_top5, # 当前epoch的top5准确率
            "timestamp": datetime.now().isoformat() # 当前时间戳
        }
        log_file.write(json.dumps(log_data) + "\n") # 将日志写入文件
    
    return epoch_loss, epoch_top1, epoch_top5, batch_index # 返回当前epoch的平均损失，top1准确率，top5准确率和batch索引

# TODO: 注释和说明
def validate_one_epoch(model, val_loader, criterion, device, save_log=False, save_dir=None, epoch=None, batch_index=0, log_file=None):
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
            
            batch_loss = running_loss / (prog_bar.n + 1)
            batch_top1 = top1_correct / total
            batch_top5 = top5_correct / total
            
            prog_bar.set_postfix({
                'loss': batch_loss,
                'top 1': batch_top1,
                'top 5': batch_top5
            })
            
            if save_log:
                log_data = {
                    "type": "batch_val",
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": batch_loss,
                    "top1": batch_top1,
                    "top5": batch_top5,
                    "timestamp": datetime.now().isoformat()
                }
                log_file.write(json.dumps(log_data) + "\n")
            batch_index += 1
    
    epoch_loss = running_loss / len(val_loader)
    epoch_top1 = top1_correct / total
    epoch_top5 = top5_correct / total
    
    if save_log:
        log_data = {
            "type": "epoch_val",
            "epoch": epoch,
            "loss": epoch_loss,
            "top1": epoch_top1,
            "top5": epoch_top5,
            "timestamp": datetime.now().isoformat()
        }
        log_file.write(json.dumps(log_data) + "\n")
    
    return epoch_loss, epoch_top1, epoch_top5, batch_index

def test_one_epoch(model, test_loader, criterion, device, save_log=False, save_dir=None, epoch=None, batch_index=0, log_file=None):
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
            
            batch_loss = running_loss / (prog_bar.n + 1)
            batch_top1 = top1_correct / total
            batch_top5 = top5_correct / total
            
            prog_bar.set_postfix({
                'loss': batch_loss,
                'top 1': batch_top1,
                'top 5': batch_top5
            })
            
            if save_log:
                log_data = {
                    "type": "batch_test",
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": batch_loss,
                    "top1": batch_top1,
                    "top5": batch_top5,
                    "timestamp": datetime.now().isoformat()
                }
                log_file.write(json.dumps(log_data) + "\n")
            batch_index += 1
    
    epoch_loss = running_loss / len(test_loader)
    epoch_top1 = top1_correct / total
    epoch_top5 = top5_correct / total
    
    if save_log:
        log_data = {
            "type": "epoch_test",
            "epoch": epoch,
            "loss": epoch_loss,
            "top1": epoch_top1,
            "top5": epoch_top5,
            "timestamp": datetime.now().isoformat()
        }
        log_file.write(json.dumps(log_data) + "\n")
    
    return epoch_loss, epoch_top1, epoch_top5, batch_index


def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_epochs, save_log=False, save_dir='log', pretrained=None):
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
        save_log: 是否保存日志
        save_dir: 保存日志的根目录
        pretrained: 预训练模型路径
    """
    best_top1 = 0.0
    best_state_dict = None
    
    if pretrained is not None:
        if os.path.exists(pretrained):  
            model.load_state_dict(torch.load(pretrained))
            print(f"加载预训练模型: {pretrained}")
        else:
            print(f"预训练模型不存在: {pretrained}")
    else:
        print("没有预训练模型")
    
    if save_log:
        # 创建保存目录
        log_index = 0
        while True:
            current_save_dir = os.path.join(save_dir, f"log_{log_index}")
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
                break
            log_index += 1
        save_dir = current_save_dir
        print(f"日志和模型将保存到: {save_dir}")
        log_file_path = os.path.join(save_dir, "training_log.json")
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    
    batch_index = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_top1, train_top5, batch_index = train_one_epoch(model, train_loader, optimizer, criterion, device, save_log, save_dir, epoch+1, batch_index, log_file)
        val_loss, val_top1, val_top5, batch_index = validate_one_epoch(model, val_loader, criterion, device, save_log, save_dir, epoch+1, batch_index, log_file)
        
        print(f"Train Loss: {train_loss:.4f}, Top 1: {train_top1:.4f}, Top 5: {train_top5:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Top 1: {val_top1:.4f}, Top 5: {val_top5:.4f}")

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_epoch = epoch + 1
            print(f"当前最佳top1准确率: {val_top1:.4f} in epoch {epoch+1}")
            best_state_dict = model.state_dict()
    last_state_dict = model.state_dict()
    
    if save_log:
        torch.save(best_state_dict, os.path.join(save_dir, f"best_model_top1:{best_top1:.4f}_in_epoch{best_epoch}.pth"))
        torch.save(last_state_dict, os.path.join(save_dir, f"last_model_top1:{val_top1:.4f}_in_epoch{epoch+1}.pth"))
    
    # 测试模型
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"加载最佳模型，进行测试")
    else:
        print("没有找到最佳模型，使用最后一次训练的模型")

    test_loss, test_top1, test_top5, batch_index = test_one_epoch(model, test_loader, criterion, device, save_log, save_dir, epoch+1, batch_index, log_file)
    print(f"Test Loss: {test_loss:.4f}, Top 1: {test_top1:.4f}, Top 5: {test_top5:.4f}")
    
    if save_log:
        log_file.close()
    
    return None


def predict(model, test_loader, device, pretrained=None):
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    model.eval()
    model.to(device)
    with torch.no_grad():
        top1_correct = 0
        top5_correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            top5_correct += torch.topk(outputs, 5, dim=1).indices.eq(labels.view(-1, 1)).sum().item()
        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total
        print(f"Top 1 Accuracy: {top1_accuracy:.4f}, Top 5 Accuracy: {top5_accuracy:.4f}")
    return top1_accuracy, top5_accuracy


def logshow(log_file_path):
    # 读取日志文件
    with open(log_file_path, 'r') as f:
        logs = [json.loads(line) for line in f]

    # 将日志转换为 DataFrame
    df = pd.DataFrame(logs)

    # 提取 epoch_train 和 epoch_val 数据
    train_data = df[df['type'] == 'epoch_train']
    val_data = df[df['type'] == 'epoch_val']
    test_data = df[df['type'] == 'epoch_test']

    # 可视化 epoch_train 和 epoch_val 的曲线图
    if not train_data.empty and not val_data.empty:
        plt.figure(figsize=(15, 5))

        # Loss 曲线
        plt.subplot(1, 3, 1)
        plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss')
        plt.plot(val_data['epoch'], val_data['loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # Top1 曲线
        plt.subplot(1, 3, 2)
        plt.plot(train_data['epoch'], train_data['top1'], label='Train Top1')
        plt.plot(val_data['epoch'], val_data['top1'], label='Val Top1')
        plt.xlabel('Epoch')
        plt.ylabel('Top1 Accuracy')
        plt.title('Top1 Accuracy Curve')
        plt.legend()

        # Top5 曲线
        plt.subplot(1, 3, 3)
        plt.plot(train_data['epoch'], train_data['top5'], label='Train Top5')
        plt.plot(val_data['epoch'], val_data['top5'], label='Val Top5')
        plt.xlabel('Epoch')
        plt.ylabel('Top5 Accuracy')
        plt.title('Top5 Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # 可视化 epoch_test 的柱状图
    if not test_data.empty:
        plt.figure(figsize=(8, 5))
        metrics = ['loss', 'top1', 'top5']
        values = [test_data.iloc[0][metric] for metric in metrics]
        plt.bar(metrics, values, color=['blue', 'green', 'orange'])
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Test Metrics')
        plt.show()