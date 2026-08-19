# Phase 6 三投影与 Projected Poincaré 差异审计

## 结论

原图差异主要来自可视化口径不一致，而不是动力学数据互相矛盾：三维图只画 `sample0`，二维图把 8 个 sample 的 crossing 混在一起；后期 checkpoint 的不同文本样本收敛到不同投影固定点，sample 间差异远大于单条轨迹的残余波动。

因此，原二维图不能用来判断“单个 sample 是否像三维图那样收敛”。它展示的是多个不同吸引终点的混合。

## 原实现问题

### 样本口径不一致

- 三维：`checkpoint × sample0` 的 256 步轨迹。
- 二维：同一 checkpoint 下 8 个 sample 的全部 crossing。
- 二维没有用颜色或分面区分 sample。

### 截面并不相同

对每个 sample $s$ 单独定义：

$$
c_s=\operatorname{median}_t z_{0,s}(t).
$$

所以 8 个 sample 的 crossing 来自 8 个不同截面 $z_0=c_s$。将它们直接叠加不能称为一个共同 Poincaré 截面。

### 未插值

原实现检测：

$$
z_0(t-1)\le c_s<z_0(t),
$$

但直接保存时刻 $t$ 的 $(z_1,z_2)$。更正后使用：

$$
\alpha=\frac{c_s-z_0(t-1)}{z_0(t)-z_0(t-1)},
$$

$$
(z_1^*,z_2^*)=(1-\alpha)(z_1,z_2)_{t-1}+\alpha(z_1,z_2)_t.
$$

## 数据证据

| checkpoint | sample0 全轨迹跨度 | sample0 尾64跨度 | Poincaré sample内 RMS | sample间 centroid 跨度 | 解释 |
|---|---:|---:|---:|---:|---|
| step0 | 5.313 | 2.778 | 0.611 | 1.717 | 尾部仍扩展，不支持收敛 |
| step1000 | 3.973 | 2.611 | 0.833 | 0.934 | 仍是非固定/未决轨迹 |
| step16000 | 5.22e-3 | 4.75e-4 | 6.37e-4 | 4.123 | 单轨道明显收缩；原二维宽度由不同 sample 终点主导 |
| step143000 | 1.72e-5 | 1.44e-5 | 2.56e-6 | 48.707 | 单轨道仅数值抖动；原二维大散点几乎完全是样本间固定点差异 |

step143000 的三维图曾因 Matplotlib 自动坐标缩放把约 `1.7e-5` 的变化铺满整个坐标框，看起来像杂乱轨迹。更正图会相对最终状态中心化并把真实 span 写入标题。

## “是否应该显示渐进周期性”

不一定。若轨迹收敛到周期为 $p$ 的极限环，合适的 Poincaré 截面通常出现有限个重复点；若轨迹收敛到固定点，它可能最终不再穿越截面，或者只因浮点抖动在固定点附近反复产生伪 crossing。step16000 的现象更接近阻尼螺旋向固定点收缩，而不是稳定极限环；step143000 是固定点附近的 float32 抖动。当前数据没有提供后期 checkpoint 渐进周期性的正证据。

## 更正后的阅读规则

- 主三维图、return map 和主 Projected Poincaré 图均只画同一个 `sample0`，可以直接互相对照；Poincaré 点由紫到黄表示 crossing 由早到晚。
- 星号是该轨迹最终投影状态，坐标使用 $\Delta z_i=z_i-z_i(T)$。
- 多 sample 图先分别减去各自最终投影状态，只回答“sample 内是否收缩”；由于截面不同，不能把它解释为一个共同 Poincaré map。
- 几何图仍需与 relative-step、nearby separation 和 Benettin Lyapunov 联合判读。
