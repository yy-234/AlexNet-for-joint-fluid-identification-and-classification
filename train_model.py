import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib使用英文
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义输出目录
OUTPUT_DIR = '/traindata2/yuying/guanjieye/model_final_5M_train/'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义6个分类的类别名称（英文）
class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']

class JointFluidDataset(Dataset):
    """关节液图像数据集"""
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file: including image_file and label 
            transform: 
        """
        self.data = []
        self.transform = transform
        
        # 读取txt文件
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split('\t')
                    if len(parts) == 2:
                        image_path = parts[0]
                        label = int(parts[1])
                        self.data.append({
                            'image_path': image_path,
                            'label': label
                        })
        
        print(f" {len(self.data)} samples")
        
        # 显示原始类别分布
        label_counts = {}
        for item in self.data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("data distribution:")
        for label in sorted(label_counts.keys()):
            print(f"  {label}: {label_counts[label]} samples")
        
        # 将标签从1-6转换为0-5（PyTorch需要从0开始）
        for item in self.data:
            item['label'] = item['label'] - 1
        
        # 统计转换后的标签分布
        converted_label_counts = {}
        for item in self.data:
            label = item['label']
            converted_label_counts[label] = converted_label_counts.get(label, 0) + 1
        
        print(f"converted label distribution: {converted_label_counts}")
        
        # 显示前几个样本
        print(f"first 5 samples:")
        for i, item in enumerate(self.data[:5]):
            print(f"  sample {i+1}: {item['image_path']} -> label {item['label']}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        
        # 加载图像
        try:
            if not os.path.exists(image_path):
                print(f"⚠️  image file not found: {image_path}")
                # 返回一个黑色图像作为占位符
                image = Image.new('RGB', (80, 80), (0, 0, 0))
            else:
                image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌  cannot load image {image_path}: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (80, 80), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
#AlexNet Model
def create_model():
    """create modified AlexNet model"""
    
    model = models.alexnet(pretrained=True)
    
    print("original AlexNet classifier structure:")
    print(model.classifier)
    
    # 修改全连接层维度
    # 将第1和第2个全连接层的维度改为512
    model.classifier[1] = nn.Linear(9216, 512)  # 第一个全连接层
    model.classifier[4] = nn.Linear(512, 512)   # 第二个全连接层
    # 将最后一个全连接层的维度改为6
    model.classifier[6] = nn.Linear(512, 6)     # 最后一个全连接层
    
    print("\nmodified AlexNet classifier structure:")
    print(model.classifier)
    
    # 验证输出维度
    print(f"\nmodel output dimension: {model.classifier[6].out_features}")
    print(f"expected output dimension: 6 (6 classes)")
    
    return model

def get_transforms():
    """get image transforms"""
    return transforms.Compose([
        transforms.Resize(100),  # 首先调整到100*100
        transforms.CenterCrop(80),  # 中心裁剪成80*80
        transforms.ToTensor(),  # 转换为张量，自动将0-255缩放到0-1
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 将0-1缩放到(-1,1)
    ])

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device=None):
    """train model"""
    if device is None:
        device = get_preferred_device()
    
    print(f"using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # record training process
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    final_metrics = {}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_predictions = []
        all_train_labels = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 收集预测结果用于分析
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * correct / total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        train_accuracies.append(train_accuracy) # 记录训练准确率
        
        # 分析预测分布
        unique_preds = sorted(list(set(all_predictions)))
        unique_labels = sorted(list(set(all_labels)))
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'training loss: {avg_train_loss:.4f}, validation loss: {avg_val_loss:.4f}')
        print(f'training accuracy: {train_accuracy:.2f}%, validation accuracy: {val_accuracy:.2f}%')
        print(f'predicted classes: {unique_preds} (total {len(unique_preds)} classes)')
        print(f'true classes: {unique_labels} (total {len(unique_labels)} classes)')
        print('-' * 50)
        
        # 保存最后一个epoch的模型
        if epoch == num_epochs - 1:
            # 计算详细指标
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            # 确保所有类别都被包含
            all_classes = list(range(6))  # 0-5
            precision = precision_score(all_labels, all_predictions, labels=all_classes, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_predictions, labels=all_classes, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, labels=all_classes, average='macro', zero_division=0)
            cm = confusion_matrix(all_labels, all_predictions, labels=all_classes)
            
            final_metrics = {
                'epoch': epoch,
                'val_accuracy': val_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': all_predictions,
                'labels': all_labels
            }
            
            model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'final_metrics': final_metrics
            }, model_path)
            print(f"save final model to: {model_path}")
            print(f"final validation accuracy: {val_accuracy:.2f}%")
            print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        
        scheduler.step()
    
    return train_losses, val_losses, val_accuracies, train_accuracies, final_metrics

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # 检查实际出现的类别
    unique_labels = sorted(list(set(all_labels + all_predictions)))
    num_classes = len(unique_labels)
    
    print(f"检测到 {num_classes} 个类别: {unique_labels}")
    print(f"预期 6 个类别: [0, 1, 2, 3, 4, 5]")
    
    # 显示每个类别的样本数量
    label_counts = {}
    pred_counts = {}
    for label in range(6):
        label_counts[label] = all_labels.count(label)
        pred_counts[label] = all_predictions.count(label)
    
    print(f"\ntrue label distribution: {label_counts}")
    print(f"predicted label distribution: {pred_counts}")
    
    # ensure using 6 classes for evaluation
    # if some classes are not in the test set, still include them in the evaluation
    expected_classes = list(range(6))  # 0-5
    
    # generate classification report, force include all 6 classes
    try:
        report = classification_report(all_labels, all_predictions, 
                                     labels=expected_classes,
                                     target_names=class_names, 
                                     output_dict=True,
                                     zero_division=0)
    except Exception as e:
        print(f"classification report generation failed: {e}")
        print("using default settings to generate report...")
        report = classification_report(all_labels, all_predictions, 
                                     output_dict=True,
                                     zero_division=0)
    
    return accuracy, report, all_labels, all_predictions, class_names

def get_preferred_device():
    """get preferred GPU device"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"detected {gpu_count} GPUs")
        
        # prefer GPU 7, then GPU 5
        preferred_gpus = [4,5,6,7]
        for gpu_id in preferred_gpus:
            if gpu_id < gpu_count:
                device = torch.device(f"cuda:{gpu_id}")
                print(f"选择GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                return device
        
        # if GPU 7 and 5 are not available, use the first available GPU
        device = torch.device("cuda:0")
        print(f"GPU 7 and 5 are not available, using GPU 0: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("no GPU detected, using CPU")
        return device

def plot_training_history(train_losses, val_losses, val_accuracies, train_accuracies=None):
    """plot training history"""
    # 设置英文字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失曲线
    axes[0,0].plot(train_losses, label='Training Loss')
    axes[0,0].plot(val_losses, label='Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training and Validation Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 绘制准确率曲线
    axes[0,1].plot(val_accuracies, label='Validation Accuracy', color='green')
    if train_accuracies:
        axes[0,1].plot(train_accuracies, label='Training Accuracy', color='blue')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].set_title('Training and Validation Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 绘制损失对比
    axes[1,0].plot(train_losses, label='Training Loss', color='blue')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].set_title('Training Loss')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 绘制验证准确率
    axes[1,1].plot(val_accuracies, label='Validation Accuracy', color='green')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].set_title('Validation Accuracy')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"training history plot saved to: {save_path}")

def main():
    # ==================== training parameters configuration ====================
    # 在这里修改训练参数
    NUM_EPOCHS = 100         # training epochs, recommended range: 20-200
    LEARNING_RATE = 0.001    # learning rate, recommended range: 0.0001-0.01
    BATCH_SIZE = 8           # batch size, recommended range: 4-16
    TRAIN_RATIO = 0.6       # training set ratio, recommended range: 0.7-0.9
    
    print(f"training parameters:")
    print(f"  training epochs: {NUM_EPOCHS}")
    print(f"  learning rate: {LEARNING_RATE}")
    print(f"  batch size: {BATCH_SIZE}")
    print(f"  training set ratio: {TRAIN_RATIO}")
    print("=" * 50)
    
    # set path
    label_txt = '/label5m.txt'
    
    # check if file exists
    if not os.path.exists(label_txt):
        print(f"错误: 找不到数据文件 {label_txt}")
        return
    
    # create dataset
    transform = get_transforms()
    train_dataset = JointFluidDataset(label_txt, transform)
    test_dataset = JointFluidDataset(label_txt, transform)
    
    # validate data
    print("\n=== 数据验证 ===")
    print(f"训练数据: {len(train_dataset)} 个样本")
    print(f"测试数据: {len(test_dataset)} 个样本")
    
    # analyze training data class distribution
    train_labels = [item['label'] for item in train_dataset.data]
    train_label_counts = {}
    for label in train_labels:
        train_label_counts[label] = train_label_counts.get(label, 0) + 1
    print(f"training data class distribution: {train_label_counts}")
    
    # 分析测试数据的类别分布
    test_labels = [item['label'] for item in test_dataset.data]
    test_label_counts = {}
    for label in test_labels:
        test_label_counts[label] = test_label_counts.get(label, 0) + 1
    print(f"test data class distribution: {test_label_counts}")
    
    print(f"training data class range: {min(train_labels)} - {max(train_labels)}")
    print(f"test data class range: {min(test_labels)} - {max(test_labels)}")
    
    # 检查图像文件是否存在
    print(f"\n=== check image file ===")
    sample_items = train_dataset.data[:5]
    for i, item in enumerate(sample_items):
        image_path = item['image_path']
        label = item['label']
        exists = os.path.exists(image_path)
        print(f"sample {i+1}: {image_path} -> label {label} -> {'✅ exists' if exists else '❌ not exists'}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
            # create model
    model = create_model()
    print("model created")
    
    # get preferred device
    device = get_preferred_device()
    
    # train model
    print("start training...")
    train_losses, val_losses, val_accuracies, train_accuracies, final_metrics = train_model(
        model, train_loader, test_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # plot training history
    plot_training_history(train_losses, val_losses, val_accuracies, train_accuracies)
    
    # load best model for evaluation
    model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"加载最终模型 (Epoch {checkpoint['epoch']+1}, 验证准确率: {checkpoint['val_accuracy']:.2f}%)")
    
    # final evaluation
    print("\n=== final model evaluation ===")
    if final_metrics:
        print(f"final Epoch: {final_metrics['epoch']+1}")
        print(f"final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"macro average precision: {final_metrics['precision']:.4f}")
        print(f"macro average recall: {final_metrics['recall']:.4f}")
        print(f"macro average F1 score: {final_metrics['f1_score']:.4f}")
        
        # 显示混淆矩阵
        print(f"\nconfusion matrix:")
        cm = final_metrics['confusion_matrix']
        for i in range(6):
            row_str = " ".join([f"{cm[i,j]:3d}" for j in range(6)])
            print(f"class {i}: {row_str}")
    
    # 保存预测结果  
    if final_metrics:
        results_df = pd.DataFrame({
            'true_label': [label + 1 for label in final_metrics['labels']],  # 转换回1-6
            'predicted_label': [pred + 1 for pred in final_metrics['predictions']],  # 转换回1-6
            'true_class': [class_names[label] for label in final_metrics['labels']],
            'predicted_class': [class_names[pred] for pred in final_metrics['predictions']]
        })
        results_path = os.path.join(OUTPUT_DIR, 'final_results.xlsx')
        results_df.to_excel(results_path, index=False)
        print(f"最终结果已保存到: {results_path}")
    
    # 保存详细报告
    report_path = os.path.join(OUTPUT_DIR, 'final_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("final evaluation report of joint fluid image classification model\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"training parameters:\n")
        f.write(f"  训练轮数training epochs: {NUM_EPOCHS}\n")
        f.write(f"  学习率learning rate: {LEARNING_RATE}\n")
        f.write(f"  批次大小: {BATCH_SIZE}\n\n")
        f.write(f"data information:\n")
        f.write(f"  总样本数total samples: {len(train_dataset)}\n")
        f.write(f"  训练样本数training samples: {len(train_dataset)}\n")
        f.write(f"  测试样本数test samples: {len(test_dataset)}\n\n")
        
        if final_metrics:
            f.write(f"最终模型性能final model performance:\n")
            f.write(f"  最终Epochfinal Epoch: {final_metrics['epoch']+1}\n")
            f.write(f"  最终验证准确率final validation accuracy: {final_metrics['val_accuracy']:.4f}\n")
            f.write(f"  宏平均精确率macro average precision: {final_metrics['precision']:.4f}\n")
            f.write(f"  宏平均召回率macro average recall: {final_metrics['recall']:.4f}\n")
            f.write(f"  宏平均F1分数macro average F1 score  : {final_metrics['f1_score']:.4f}\n\n")
            
            f.write("confusion matrix:\n")
            cm = final_metrics['confusion_matrix']
            f.write("预测标签 →\n")
            f.write("真实标签 ↓\n")
            f.write("          " + " ".join([f"{name:>8}" for name in class_names]) + "\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:>10}")
                for j in range(6):
                    f.write(f"{cm[i, j]:>8}")
                f.write("\n")
    
    print(f"detailed evaluation report saved to: {report_path}")
    
    # 保存指标到Excel
    if final_metrics:
        metrics_df = pd.DataFrame({
            '指标': ['验证准确率', '宏平均精确率', '宏平均召回率', '宏平均F1分数'],
            '值': [final_metrics['val_accuracy'], final_metrics['precision'], 
                   final_metrics['recall'], final_metrics['f1_score']]
        })
        metrics_path = os.path.join(OUTPUT_DIR, 'final_metrics.xlsx')
        metrics_df.to_excel(metrics_path, index=False)
        print(f"最终指标已保存到: {metrics_path}")
    
    print("\n=== 训练完成 ===")
    print("生成的文件:")
    print(f"- {os.path.join(OUTPUT_DIR, 'final_model.pth')}: 最终模型")
    print(f"- {os.path.join(OUTPUT_DIR, 'training_history.png')}: 训练历史图")
    print(f"- {os.path.join(OUTPUT_DIR, 'final_results.xlsx')}: 最终预测结果")
    print(f"- {os.path.join(OUTPUT_DIR, 'final_evaluation_report.txt')}: 详细评估报告")
    print(f"- {os.path.join(OUTPUT_DIR, 'final_metrics.xlsx')}: 最终指标汇总")

if __name__ == "__main__":
    main() 