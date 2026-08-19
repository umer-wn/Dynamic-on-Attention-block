# 阶段六：训练 checkpoint 临界性

## 1. 研究问题

Pythia-70M 的语言建模改善是否伴随当前 direct embedding-feedback 算子向零 Lyapunov 临界点移动？稳定点、慢稳定、未决与混沌候选能否被区分？

## 2. 实验和数据来源

比较 step0、step1000、step16000、step143000；主协议使用同一批 8 个 WikiText 样本、seq64、burn512、eval256。长程数据位于 `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality/long_asymptotic/`，三投影复测位于 `visualization_rerun/`。

## 3. 核心参数

direct operator、epsilon=`1e-3`、8 samples、seq64、burn512、eval256；投影 bank 使用 fixed random、count=3、seed=1234、跨样本共享。复测不重复计算 Frobenius、Lyapunov 或历史 product metric。

这里的 `step0/1000/16000/143000` 是加载的 Pythia 官方训练 checkpoint，不是本实验重新训练了四个模型。对每个 checkpoint，模型权重始终冻结；真实文本只用于得到初始 token embedding $x_0$。之后执行的是静止权重下的连续状态循环：

$$
x_{t+1}=F_{\theta_s}(x_t)=H_L(x_t;\theta_s),
$$

即把第 $t$ 次前向传播的最后层 hidden state 作为第 $t+1$ 次前向传播的 `inputs_embeds`。这不是逐 token 生成，也没有 loss、反向传播或 optimizer update。

主 Lyapunov/Frobenius 协议实际使用 burn512、eval128、4 个 Frobenius states、每状态 4 个 Rademacher probes、2 个 Lyapunov probes。三投影复测使用 burn512、eval256，但把上述 probes 设为 0，只补采轨迹几何。报告中“训练 step”与“eval step”不可互换：前者选择 $\theta_s$，后者是固定 $\theta_s$ 下的动力学时间 $t$。

关键公式为：

$$
r_t=\frac{\|x_t-x_{t-1}\|_2}{\max(\|x_t\|_2,10^{-12})},
$$

$$
\rho_t\approx\frac{\|J_t\|_F}{\sqrt D}
=\sqrt{\frac{\mathbb E_v\|J_tv\|_2^2}{D}},
$$

$$
\widehat\lambda_{\max}=\frac1T\sum_t\log\|J_tv_t\|_2,
\qquad v_{t+1}=\frac{J_tv_t}{\|J_tv_t\|_2}.
$$

因此，Frobenius 是所有奇异方向的 RMS 增益，而 Lyapunov 跟踪沿轨道逐渐对齐的主切向方向；二者本来就不要求符号等价。

## 4. 图表

![性能](../assets/experiment_visualization/phase_06__checkpoint-performance.png)

![样本级 Lyapunov](../assets/experiment_visualization/phase_06__checkpoint-lyapunov-samples.png)

![短长 burn-in](../assets/experiment_visualization/phase_06__short-vs-long-protocol.png)

![relative step](../assets/experiment_visualization/phase_06__checkpoint-relative-step-trajectories.png)

![recurrence](../assets/experiment_visualization/phase_06__checkpoint-recurrence-heatmap.png)

![nearby separation](../assets/experiment_visualization/phase_06__checkpoint-nearby-trajectories.png)

![三投影轨迹](../assets/experiment_visualization/phase_06__checkpoint-three-projection-trajectories.png)

![return maps](../assets/experiment_visualization/phase_06__checkpoint-return-maps.png)

![Projected Poincaré Sections](../assets/experiment_visualization/phase_06__checkpoint-projected-poincare.png)

![各 sample 分别中心化的 Projected Poincaré 诊断](../assets/experiment_visualization/phase_06__checkpoint-poincare-per-sample-centered.png)

## 5. 结果解释

训练显著降低 loss/PPL，但当前算子没有随训练向零 Lyapunov 单调靠近。step0 为正 Lyapunov 和强扰动扩张的混沌候选；step1000 均值略负但有限扰动仍有长暂态，标为临界附近未决；step16000 为慢收敛稳定候选；step143000 具有强负 Lyapunov、扰动收缩和 numerical-scale relative delta，判为稳定定点。

三投影使用完全相同的随机方向，使 checkpoint 之间的几何比较可复核。更正后的三维、return map 和主 Projected Poincaré 图都只画相同的 `sample0`，并相对该轨迹最终投影状态中心化。Projected Poincaré 只在 `z0` 向上穿越该 sample 投影中位数时记录线性插值后的 `(z1,z2)`；颜色由紫到黄表示 crossing 由早到晚。step16000 的 sample0 全轨迹跨度约 `5.22e-3`，尾 64 步缩至 `4.75e-4`，支持阻尼式定点收敛；step143000 的全跨度仅约 `1.72e-5`，crossing 的 sample 内 RMS 约 `2.56e-6`，是固定点附近的数值抖动而非周期轨道。

原图曾把 8 个 sample 的 crossing 直接叠加，而每个 sample 又使用各自的 `median(z0)` 截面。step143000 的 sample 间截面 centroid 跨度约 `48.7`，远大于 sample 内 `2.56e-6`，所以宽散点主要代表不同文本样本收敛到不同固定点，不代表单条轨迹发散。新增的多 sample 图先减去各自最终投影状态，只作为 sample 内收敛诊断，不宣称为共同 Poincaré 截面。完整审计见 [Phase 6 三投影与 Projected Poincaré 差异审计](../phase6_poincare_projection_audit.md)。

逐图阅读顺序建议如下：

1. 先看 performance 图，只确认 checkpoint 的语言建模性能随官方训练进度改善。
2. 再看样本级 Lyapunov 图，以零线区分有限时间主切向扩张和收缩，并观察 8 个样本的离散度。
3. 用 Frobenius–Lyapunov 散点检查平均增益 proxy；step0 位于 `Frobenius<1、Lyapunov>0` 象限，是最重要的反例。
4. 用 short-vs-long 与 relative-step 图排除暂态，并区分固定点、慢收敛和非固定轨道。
5. 用 nearby distance 验证有限扰动的方向，但不覆盖 Benettin Lyapunov 结论。
6. 最后才看 recurrence、3D 投影、return map 和 Projected Poincaré；这些图解释几何形态，不能单独给出严格相位标签。

## 6. 已发现缺陷与更正

step0 的 `Frobenius<1` 与正 Lyapunov 直接否定了把平均增益当相位标签的做法。短 burn-in 会混入长暂态；固定点收敛不能替代混沌诊断；单投影 Poincaré 不能代表完整吸引子。原 Poincaré 图还存在 sample0 三维轨迹与 8-sample 二维叠加口径不一致、不同 sample 截面混画和 crossing 未插值的问题，现已更正。

## 7. 当前能证明、不能证明的假设

能支持：当前 direct operator 随训练由早期扩张转向后期稳定；step143000 是稳定定点；Frobenius proxy 在 step0 失效。不能最终证明：step0 已满足所有严格混沌条件；step1000 恰在相变点；训练普遍让 LLM 远离临界；原生 LLM 生成动力学的相位与本闭环相同。

## 8. 下一阶段为什么发生

应增加 checkpoint 密度、样本和 projection/Lyapunov seeds，报告窗口收敛和置信区间，并把 direct operator 与理论动机更强的替代闭环、surrogate controls 同时预注册比较。
