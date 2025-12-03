"""
数据可视化工具脚本
包含训练过程中的各种可视化函数
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='indicator'):
    """
    绘制训练和验证的损失曲线、准确率曲线
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 创建2x1子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制损失曲线
    axes[0].plot(epochs, train_losses, 'b-o', label='训练损失', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='验证损失', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    axes[1].plot(epochs, train_accs, 'b-o', label='训练准确率', linewidth=2, markersize=6)
    axes[1].plot(epochs, val_accs, 'r-s', label='验证准确率', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 训练曲线已保存到: {os.path.join(save_path, 'training_curves.png')}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='indicator'):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    
    plt.title('混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=13)
    plt.ylabel('真实标签', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 混淆矩阵已保存到: {os.path.join(save_path, 'confusion_matrix.png')}")


def plot_classification_report(y_true, y_pred, class_names, save_path='indicator'):
    """
    生成并保存分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 生成分类报告
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names, 
                                   digits=4)
    
    # 保存为文本文件
    report_path = os.path.join(save_path, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('分类报告\n')
        f.write('=' * 80 + '\n\n')
        f.write(report)
    
    print(f"✓ 分类报告已保存到: {report_path}")
    print("\n分类报告:")
    print(report)


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path='indicator'):
    """
    绘制每个类别的准确率柱状图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的准确率
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    
    # 绘制柱状图
    bars = plt.bar(range(len(class_names)), per_class_acc, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # 在柱子上添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('各类别识别准确率', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 各类别准确率图已保存到: {os.path.join(save_path, 'per_class_accuracy.png')}")


def save_training_log(config, train_losses, train_accs, val_losses, val_accs, 
                      best_acc, best_epoch, save_path='indicator'):
    """
    保存训练日志
    
    Args:
        config: 配置字典
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        best_acc: 最佳验证准确率
        best_epoch: 最佳epoch
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    log_path = os.path.join(save_path, 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('训练日志\n')
        f.write('=' * 80 + '\n\n')
        
        # 写入配置信息
        f.write('训练配置:\n')
        f.write('-' * 80 + '\n')
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
        f.write('\n')
        
        # 写入最佳结果
        f.write('最佳结果:\n')
        f.write('-' * 80 + '\n')
        f.write(f'最佳验证准确率: {best_acc:.4f}%\n')
        f.write(f'最佳Epoch: {best_epoch}\n')
        f.write('\n')
        
        # 写入每个epoch的详细信息
        f.write('训练详情:\n')
        f.write('-' * 80 + '\n')
        f.write(f"{'Epoch':<8}{'Train Loss':<15}{'Train Acc':<15}{'Val Loss':<15}{'Val Acc':<15}\n")
        f.write('-' * 80 + '\n')
        
        for i in range(len(train_losses)):
            f.write(f"{i+1:<8}{train_losses[i]:<15.4f}{train_accs[i]:<15.2f}"
                   f"{val_losses[i]:<15.4f}{val_accs[i]:<15.2f}\n")
    
    print(f"✓ 训练日志已保存到: {log_path}")


def plot_learning_rate_schedule(lr_history, save_path='indicator'):
    """
    绘制学习率变化曲线
    
    Args:
        lr_history: 学习率历史记录列表
        save_path: 保存路径
    """
    if not lr_history:
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('学习率变化曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 学习率曲线已保存到: {os.path.join(save_path, 'learning_rate_schedule.png')}")


def visualize_all(y_true, y_pred, class_names, train_losses, train_accs, 
                  val_losses, val_accs, config, best_acc, best_epoch, 
                  lr_history=None, save_path='indicator'):
    """
    一次性生成所有可视化图表
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        config: 配置字典
        best_acc: 最佳验证准确率
        best_epoch: 最佳epoch
        lr_history: 学习率历史记录
        save_path: 保存路径
    """
    print("\n" + "="*80)
    print("开始生成可视化结果...")
    print("="*80)
    
    # 生成各种图表
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path)
    plot_confusion_matrix(y_true, y_pred, class_names, save_path)
    plot_classification_report(y_true, y_pred, class_names, save_path)
    plot_per_class_accuracy(y_true, y_pred, class_names, save_path)
    save_training_log(config, train_losses, train_accs, val_losses, val_accs, 
                     best_acc, best_epoch, save_path)
    
    if lr_history:
        plot_learning_rate_schedule(lr_history, save_path)
    
    print("="*80)
    print("所有可视化结果已生成完成！")
    print("="*80 + "\n")
