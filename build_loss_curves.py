# -*- coding: utf-8 -*-
"""绘制 0825_01 / 0825_02 两阶段(对比学习+监督学习) loss 曲线"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 对比学习 loss (epoch 1-10)
contrastive_01 = [4.4945, 4.4217, 4.4095, 4.4019, 4.3970, 4.3925, 4.3907, 4.3900, 4.3894, 4.3874]
contrastive_02 = [4.6764, 4.6190, 4.6073, 4.6024, 4.6046, 4.5929, 4.5949, 4.5895, 4.5957, 4.5923]

# 监督学习 loss (epoch 1-20 / 1-7)
supervised_01 = [0.0708, 0.0593, 0.0572, 0.0559, 0.0550, 0.0547, 0.0537, 0.0534, 0.0532, 0.0526,
                 0.0526, 0.0525, 0.0518, 0.0518, 0.0521, 0.0517, 0.0514, 0.0512, 0.0512, 0.0507]
supervised_02 = [0.3123, 0.2980, 0.2937, 0.2918, 0.2901, 0.2895, 0.2894]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# 左：对比学习
ax = axes[0]
ax.plot(range(1, 11), contrastive_01, 'o-', color='#2b83ba', label='0825_01 (内部85/15)')
ax.plot(range(1, 11), contrastive_02, 's--', color='#d7191c', label='0825_02 (外部valid)')
ax.set_xlabel('对比学习 Epoch')
ax.set_ylabel('Contrastive Loss')
ax.set_title('Phase 1 对比学习 Loss')
ax.legend()
ax.grid(alpha=0.3)

# 右：监督学习
ax = axes[1]
ax.plot(range(1, 21), supervised_01, 'o-', color='#2b83ba', label='0825_01 (内部85/15)')
ax.plot(range(1, 8), supervised_02, 's--', color='#d7191c', label='0825_02 (外部valid, 早停 ep7)')
ax.set_xlabel('监督学习 Epoch')
ax.set_ylabel('Supervised Loss')
ax.set_title('Phase 2 监督学习 Loss')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curves_0825.png', dpi=150, bbox_inches='tight')
print('saved: loss_curves_0825.png')
