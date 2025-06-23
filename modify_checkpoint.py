import torch

# 参数
checkpoint_path = './res/cp/80000_iter.pth'
new_batch_size = 32
old_batch_size = 16

# 加载 checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

# 读取旧的 optimizer 状态
optim_state = checkpoint['optim']

# 计算新的学习率（线性放缩）
for param_group in optim_state['param_groups']:
    old_lr = param_group['lr']
    new_lr = old_lr * (new_batch_size / old_batch_size)
    param_group['lr'] = new_lr
    print(f"调整学习率: {old_lr} -> {new_lr}")

# 保存新 checkpoint
new_checkpoint_path = './res/cp/80000_iter.pth'
torch.save(checkpoint, new_checkpoint_path)
print(f"新 checkpoint 已保存到 {new_checkpoint_path}")
