"""
训练 Lena-TRNN 模型，验证能量分数与预测误差的正相关性
修改：训练时直接使用模型前向，不进行能量更新迭代
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import traceback

try:
    print("正在导入所需库...")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from model import Lena_TRNN
    from data_utils import get_dataloaders
    print("库导入成功。")
except Exception as e:
    print("导入库时出错：", e)
    traceback.print_exc()
    sys.exit(1)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 超参数
INPUT_DIM = 7
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
PRED_LEN = 96
INPUT_LEN = 96
ENERGY_STEPS = 3   # 推理时用的步数
ALPHA = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("加载数据...")
try:
    train_loader, val_loader, test_loader, data_mean, data_std = get_dataloaders(
        batch_size=BATCH_SIZE, input_len=INPUT_LEN, pred_len=PRED_LEN
    )
    print("数据加载成功。")
except Exception as e:
    print("数据加载失败：", e)
    traceback.print_exc()
    sys.exit(1)

print("初始化模型...")
try:
    model = Lena_TRNN(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print("模型初始化成功。")
except Exception as e:
    print("模型初始化失败：", e)
    traceback.print_exc()
    sys.exit(1)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss()

# 训练
print("开始训练...")
epoch_losses = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        batch_size = x.size(0)
        # 构造初始预测：历史 x + 未来全零
        init_pred = torch.cat([x, torch.zeros(batch_size, PRED_LEN, INPUT_DIM).to(device)], dim=1)

        # 直接使用模型前向，不进行能量更新
        pred_full = model(init_pred)  # (b, 192, 7)
        pred_future = pred_full[:, -PRED_LEN:, :]  # (b, 96, 7)

        loss = criterion(pred_future, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, 训练损失: {avg_loss:.4f}")

print("训练完成。")

# 可视化（推理模式，使用能量更新）
print("开始生成可视化图像...")
model.eval()

try:
    x_sample, y_sample = next(iter(test_loader))
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)
    batch_size = x_sample.size(0)
    init_pred = torch.cat([x_sample, torch.zeros(batch_size, PRED_LEN, INPUT_DIM).to(device)], dim=1)

    # 移除 torch.no_grad()，让 energy_update 内部处理梯度
    all_preds, all_energies = model.energy_update(
        init_pred, steps=ENERGY_STEPS, alpha=ALPHA, return_all=True
    )

    sample_idx = 0
    var_idx = 0

    true_history = x_sample[sample_idx].cpu().numpy()
    true_future = y_sample[sample_idx].cpu().numpy()
    true_full = np.concatenate([true_history[:, var_idx], true_future[:, var_idx]], axis=0)
    time_axis = np.arange(192)

    # 图1：多步预测演化
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, true_full, 'k-', linewidth=2, label='Ground Truth')

    init_pred_np = init_pred[sample_idx, :, var_idx].cpu().numpy()
    plt.plot(time_axis, init_pred_np, '--', color='gray', label='Initial (step0)')

    colors = ['b', 'g', 'r']
    for step in range(ENERGY_STEPS):
        pred_step = all_preds[step][sample_idx, :, var_idx].cpu().numpy()
        plt.plot(time_axis, pred_step, '--', color=colors[step], label=f'Pred step{step+1}')

    plt.axvline(x=95, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.title(f'Prediction evolution with energy updates (variable {var_idx})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_evolution.png', dpi=150)
    print("已保存 prediction_evolution.png")

    # 图2：能量 vs 误差
    final_pred_full = all_preds[-1][sample_idx].cpu().numpy()
    error = np.abs(final_pred_full[:, var_idx] - true_full)
    energy_final = all_energies[-1][sample_idx].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(time_axis, error, 'b-', label='Prediction Error')
    plt.xlabel('Time step')
    plt.ylabel('Error')
    plt.title('Prediction Error (MAE)')

    plt.subplot(1,2,2)
    plt.plot(time_axis, energy_final, 'r-', label='Energy Score')
    plt.xlabel('Time step')
    plt.ylabel('Energy')
    plt.title('Energy Score (last step)')
    plt.tight_layout()
    plt.savefig('energy_vs_error.png', dpi=150)
    print("已保存 energy_vs_error.png")

    # 相关系数
    corr = np.corrcoef(error, energy_final)[0,1]
    print(f"误差与能量分数的相关系数: {corr:.3f}")

    plt.figure(figsize=(5,5))
    plt.scatter(energy_final, error, alpha=0.5)
    plt.xlabel('Energy Score')
    plt.ylabel('Prediction Error')
    plt.title(f'Correlation: {corr:.3f}')
    plt.savefig('scatter_energy_error.png', dpi=150)
    print("已保存 scatter_energy_error.png")

    # 如果需要显示图像，取消下一行注释
    # plt.show()

except Exception as e:
    print("可视化过程中出错：", e)
    traceback.print_exc()
    sys.exit(1)

print("所有步骤完成。")