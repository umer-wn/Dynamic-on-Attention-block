# 实验16报告：词频分层的窗口动力学

## 1. 协议

- checkpoint：step5000, step7000, step8000, step9000, step10000, step13000, step21000, step25000, step29000, step33000, step37000, step41000, step53000, step57000, step61000。
- dynamic step：0–1024；窗口宽度 256；端点 [256, 512, 768, 1024]。
- 4个token来自分离的WikiText-2词频bin；所有checkpoint使用完全相同的token。
- Projection 1–4来自同一个固定随机正交基；CSV与JSONL均保存`(checkpoint, dynamicstep, projection(1,2,3,4))`。
- Lyapunov使用JVP传播单个扰动向量，并在每一步重新归一化。`λ=(1/T)Σ log(||J_t v_t||₂)`，单位是每个dynamic step的自然对数增长率；`λ>0`表示该有限时间轨道附近扰动平均增长，`λ<0`表示平均衰减。

### 起始token

| token | id | WikiText count | frequency bin |
|---|---:|---:|---:|
| ' clones' | 21825 | 2 | 0 |
| ' motive' | 23778 | 8 | 2 |
| ' cabinet' | 19211 | 33 | 5 |
| ' miles' | 6574 | 404 | 7 |

## 2. Checkpoint loss

loss沿用实验11的固定The Pile test协议：512个固定样本、sequence length 64、token-weighted causal cross entropy。

| checkpoint | loss |
|---|---:|
| step5000 | 3.7285567 |
| step7000 | 3.6601702 |
| step8000 | 3.6450676 |
| step9000 | 3.6348848 |
| step10000 | 3.6285040 |
| step13000 | 3.6003986 |
| step21000 | 3.5659667 |
| step25000 | 3.5724779 |
| step29000 | 3.5334199 |
| step33000 | 3.5330598 |
| step37000 | 3.5226762 |
| step41000 | 3.5434085 |
| step53000 | 3.5235889 |
| step57000 | 3.5225042 |
| step61000 | 3.5334855 |

## 3. 0–1024有限时间最大Lyapunov指数

下表中位数及范围来自4个词频分层初始token。`exp(λ)`是每一步的几何平均扰动倍率；这不是某一个端点Jacobian的谱半径。

| checkpoint | median λ | token range | exp(median λ) |
|---|---:|---:|---:|
| step5000 | 0.00908062 | [0.00655344, 0.0123562] | 1.00912 |
| step7000 | 0.0194334 | [0.00564718, 0.0204647] | 1.01962 |
| step8000 | 0.0129018 | [0.00377287, 0.0193791] | 1.01299 |
| step9000 | -0.00416657 | [-0.0049005, -0.00308506] | 0.995842 |
| step10000 | 0.0022684 | [0.00134967, 0.00371316] | 1.00227 |
| step13000 | -0.0460253 | [-0.0491874, -0.0452638] | 0.955018 |
| step21000 | 0.0221142 | [0.0153319, 0.0308317] | 1.02236 |
| step25000 | 0.0157271 | [0.0144881, 0.0189005] | 1.01585 |
| step29000 | 0.00200839 | [0.000752018, 0.00257853] | 1.00201 |
| step33000 | 0.0195711 | [0.0118511, 0.0236714] | 1.01976 |
| step37000 | 0.00750224 | [0.00340631, 0.0122954] | 1.00753 |
| step41000 | 0.00375704 | [0.0029909, 0.00427378] | 1.00376 |
| step53000 | 0.00318509 | [0.00177225, 0.0060585] | 1.00319 |
| step57000 | 0.00191671 | [0.00155982, 0.00228507] | 1.00192 |
| step61000 | -0.103038 | [-0.10335, -0.102268] | 0.902093 |

### 3.1 最后256步（768–1024）Lyapunov统计

统计样本是每个checkpoint的4个词频分层token。总体方差使用分母 `N=4`；样本方差使用分母 `N-1=3`。上下界是4个观测值的min/max，不是置信区间。

| checkpoint | mean | population variance | sample variance | lower=min | upper=max |
|---|---:|---:|---:|---:|---:|
| step5000 | 0.003311808 | 2.193676e-05 | 2.924901e-05 | -0.002648338 | 0.008718926 |
| step7000 | 0.01073478 | 0.0001211354 | 0.0001615139 | -0.006998163 | 0.02096859 |
| step8000 | 0.01009304 | 2.530297e-05 | 3.37373e-05 | 0.003835758 | 0.01706144 |
| step9000 | -0.005568249 | 5.761264e-08 | 7.681685e-08 | -0.00590705 | -0.005310655 |
| step10000 | -7.164195e-05 | 1.324191e-07 | 1.765588e-07 | -0.0004656008 | 0.000295607 |
| step13000 | -0.04956592 | 1.647987e-17 | 2.197316e-17 | -0.04956592 | -0.04956591 |
| step21000 | 0.01987953 | 8.200102e-05 | 0.0001093347 | 0.004968822 | 0.02943693 |
| step25000 | 0.01075313 | 2.353236e-05 | 3.137648e-05 | 0.004450292 | 0.01802618 |
| step29000 | -0.001096149 | 1.502108e-07 | 2.00281e-07 | -0.001490103 | -0.0006684746 |
| step33000 | 0.01788143 | 1.094824e-05 | 1.459766e-05 | 0.014225 | 0.02291282 |
| step37000 | 0.006551354 | 3.649926e-05 | 4.866568e-05 | -0.001652851 | 0.01538555 |
| step41000 | -0.0001881497 | 3.217161e-07 | 4.289548e-07 | -0.001102088 | 0.0004331879 |
| step53000 | -0.0004890024 | 1.696757e-07 | 2.262342e-07 | -0.001014856 | -2.432542e-05 |
| step57000 | 0.001556578 | 5.514537e-06 | 7.352717e-06 | -0.0002685173 | 0.005476748 |
| step61000 | -0.1054719 | 8.967115e-12 | 1.195615e-11 | -0.105477 | -0.1054695 |

### 3.2 指定checkpoint的动态周期估计

周期在完整512维hidden state上估计，不使用投影坐标。分析区间是`dynamic step 512–1024`，候选整数时滞为1–128，因此每个候选周期至少可观察4次。距离用轨道pairwise-distance P95归一化；`normalized P95`越接近0，跨周期闭合越好。

`quadratic period`是在最佳整数时滞及其左右邻点上进行抛物线插值，只用于给出亚步近似。实验14预注册阈值为：strict `≤1e-4`、approximate `≤1e-2`。未通过阈值时仅称为主回归时滞，不能据此确认严格极限环。

| checkpoint | integer period | quadratic period | token period range | normalized P95 range | absolute P95 range | classification |
|---|---:|---:|---:|---:|---:|---|
| step9000 | 22 | 21.5566 | [22, 22] | [0.1368, 0.14026] | [0.017513, 0.087342] | dominant_recurrence_only |
| step29000 | 100 | 99.9568 | [100, 100] | [0.010527, 0.010551] | [0.30916, 0.30928] | dominant_recurrence_only |
| step41000 | 57 | 56.9164 | [57, 57] | [0.016108, 0.016252] | [1.1758, 1.1843] | dominant_recurrence_only |
| step53000 | 75 | 74.9168 | [75, 75] | [0.019732, 0.18493] | [1.5116, 14.192] | dominant_recurrence_only |
| step57000 | 101 | 101.4547 | [101, 101] | [0.063767, 0.065603] | [6.0183, 6.2269] | dominant_recurrence_only |

## 4. Jacobian四个尺度

- `spectral radius = max|λᵢ|`：取特征值的模，反映局部渐近模态尺度。
- `spectral abscissa = max Re(λᵢ)`：取特征值实部；伴随列记录该特征值的虚部和模，不能用它替代离散系统的谱半径。
- `operator norm = σ₁`：最大奇异值，无实部或虚部，表示单步最强扰动放大。
- `Frobenius norm = sqrt(ΣᵢⱼJᵢⱼ²)=sqrt(Σₖσₖ²)`：所有Jacobian元素平方和开根号，也等于全部奇异值平方和开根号；它衡量整体线性响应能量，不是最强单一方向的放大率。

| checkpoint | spectral radius range | max Re range | operator norm range | Frobenius norm range | norm/radius range |
|---|---:|---:|---:|---:|---:|
| step5000 | [0.98573, 1.0353] | [0.97962, 1.0353] | [1.7213, 2.1216] | [15.236, 15.542] | [1.6978, 2.1523] |
| step7000 | [0.98139, 1.086] | [0.9733, 1.0832] | [1.8509, 2.3174] | [15.1, 16.509] | [1.7982, 2.1714] |
| step8000 | [0.95322, 1.1712] | [0.95322, 1.1554] | [1.7625, 2.2681] | [15.142, 17.516] | [1.7781, 2.1087] |
| step9000 | [0.98073, 1.0037] | [0.96075, 0.97406] | [1.9543, 1.9966] | [15.637, 15.8] | [1.9535, 2.0358] |
| step10000 | [0.95807, 1.039] | [0.92631, 0.99202] | [1.7396, 1.8679] | [15.003, 15.906] | [1.743, 1.8519] |
| step13000 | [0.95164, 0.95165] | [0.95164, 0.95165] | [1.8363, 1.8363] | [15.174, 15.174] | [1.9296, 1.9296] |
| step21000 | [0.97508, 1.1648] | [0.97232, 1.1648] | [1.7306, 4.4073] | [15.199, 17.192] | [1.672, 4.0887] |
| step25000 | [0.98751, 1.0758] | [0.98751, 1.0758] | [1.7259, 3.0081] | [14.816, 16.463] | [1.7428, 2.9335] |
| step29000 | [0.9395, 1.0404] | [0.93197, 1.0253] | [1.7488, 1.872] | [14.827, 15.723] | [1.7859, 1.8614] |
| step33000 | [0.98168, 1.0998] | [0.96984, 1.0977] | [1.8551, 2.3772] | [15.236, 16.631] | [1.8717, 2.1614] |
| step37000 | [0.98445, 1.0484] | [0.98445, 1.0453] | [2.1484, 2.3825] | [15.372, 16.263] | [2.1585, 2.3541] |
| step41000 | [0.97843, 1.1252] | [0.97808, 1.1252] | [1.8715, 4.4625] | [14.821, 16.094] | [1.8805, 4.0839] |
| step53000 | [0.9514, 1.1186] | [0.9514, 1.107] | [1.7662, 5.1448] | [14.034, 15.769] | [1.8239, 4.9098] |
| step57000 | [0.93936, 1.1739] | [0.93936, 1.1739] | [1.729, 6.2563] | [13.09, 15.611] | [1.8407, 5.3296] |
| step61000 | [0.8999, 0.89991] | [0.8999, 0.89991] | [1.9082, 1.9083] | [13.383, 13.383] | [2.1205, 2.1206] |

Jacobian图提供两种组织方式：`by_window`以每个checkpoint为子图、横轴为dynamic step；`by_checkpoint`以training checkpoint为横轴、每条线对应一个窗口端点。中心线是4个token的中位数，阴影是min–max。

- [`spectral_radius_by_checkpoint.png`](figures/spectral_radius_by_checkpoint.png)
- [`spectral_abscissa_by_checkpoint.png`](figures/spectral_abscissa_by_checkpoint.png)
- [`operator_norm_2_by_checkpoint.png`](figures/operator_norm_2_by_checkpoint.png)
- [`jacobian_frobenius_norm_by_checkpoint.png`](figures/jacobian_frobenius_norm_by_checkpoint.png)
- [`jacobian_three_metrics_by_checkpoint.png`](figures/jacobian_three_metrics_by_checkpoint.png)
- [`jacobian_four_metrics_by_checkpoint.png`](figures/jacobian_four_metrics_by_checkpoint.png)
- [`jacobian_three_metrics_with_loss.csv`](processed/jacobian_three_metrics_with_loss.csv)
- [`jacobian_four_metrics_with_loss.csv`](processed/jacobian_four_metrics_with_loss.csv)

## 5. 三类最近词与可信度

- cosine：input embedding方向相似度；可信度为top1-top2 similarity margin。
- Euclidean：input embedding绝对距离；可信度为`(d2-d1)/d1`。
- LM-head：输出头logit/softmax的预测；可信度同时保存概率差与logit差。

### 5.1 Cosine最近词

| checkpoint | dynamic step | ' clones' | ' motive' | ' cabinet' | ' miles' |
|---|---:|---|---|---|---|
| step5000 | 256 | 'GAA'<br>sim=0.20412; margin=0.0141; freq=0 | 'GAA'<br>sim=0.20821; margin=0.0257; freq=0 | 'GAA'<br>sim=0.2134; margin=0.0238; freq=0 | 'GAA'<br>sim=0.20922; margin=0.00477; freq=0 |
| step5000 | 512 | 'GAA'<br>sim=0.20102; margin=0.0194; freq=0 | 'GAA'<br>sim=0.19413; margin=0.0115; freq=0 | 'GAA'<br>sim=0.18538; margin=0.00109; freq=0 | ' maintaining'<br>sim=0.19731; margin=0.0184; freq=42 |
| step5000 | 768 | 'GAA'<br>sim=0.20057; margin=0.0132; freq=0 | 'GAA'<br>sim=0.18479; margin=0.00285; freq=0 | 'GAA'<br>sim=0.21599; margin=0.0237; freq=0 | 'GAA'<br>sim=0.2022; margin=0.00821; freq=0 |
| step5000 | 1024 | 'GAA'<br>sim=0.21928; margin=0.0259; freq=0 | 'GAA'<br>sim=0.2231; margin=0.0298; freq=0 | 'GAA'<br>sim=0.20587; margin=0.00166; freq=0 | 'GAA'<br>sim=0.19237; margin=0.0139; freq=0 |
| step7000 | 256 | ' preserv'<br>sim=0.19462; margin=0.015; freq=2 | 'ynamic'<br>sim=0.1951; margin=0.0188; freq=1 | ' repertoire'<br>sim=0.18631; margin=0.00198; freq=27 | ' politique'<br>sim=0.18656; margin=0.00881; freq=0 |
| step7000 | 512 | 'hospital'<br>sim=0.19105; margin=0.0127; freq=3 | 'hospital'<br>sim=0.18223; margin=0.00159; freq=3 | 'コ'<br>sim=0.19154; margin=0.00839; freq=2 | 'icrobial'<br>sim=0.17011; margin=0.00084; freq=0 |
| step7000 | 768 | ' politique'<br>sim=0.19542; margin=0.0176; freq=0 | 'hospital'<br>sim=0.1829; margin=0.00603; freq=3 | 'コ'<br>sim=0.19398; margin=0.0113; freq=2 | ' politique'<br>sim=0.2065; margin=0.0127; freq=0 |
| step7000 | 1024 | 'avirus'<br>sim=0.18326; margin=0.0103; freq=2 | ' prevalent'<br>sim=0.19279; margin=0.00449; freq=20 | 'コ'<br>sim=0.19451; margin=0.0115; freq=2 | ' shooting'<br>sim=0.17397; margin=0.000723; freq=82 |
| step8000 | 256 | '性'<br>sim=0.18276; margin=0.00642; freq=0 | 'Tour'<br>sim=0.18471; margin=0.00873; freq=1 | ' retrieving'<br>sim=0.17275; margin=0.00118; freq=1 | 'Ip'<br>sim=0.1649; margin=0.00292; freq=0 |
| step8000 | 512 | ' CBC'<br>sim=0.17078; margin=0.00233; freq=2 | 'Tour'<br>sim=0.16564; margin=0.000164; freq=1 | 'Tour'<br>sim=0.19327; margin=0.0254; freq=1 | ' habit'<br>sim=0.18916; margin=0.00858; freq=26 |
| step8000 | 768 | '性'<br>sim=0.16276; margin=0.000765; freq=0 | 'edom'<br>sim=0.17008; margin=0.00294; freq=0 | 'edom'<br>sim=0.17031; margin=0.00257; freq=0 | ' libert'<br>sim=0.20054; margin=0.0274; freq=2 |
| step8000 | 1024 | ' retrieving'<br>sim=0.17699; margin=0.000358; freq=1 | '性'<br>sim=0.16641; margin=0.00484; freq=0 | '性'<br>sim=0.16501; margin=0.000439; freq=0 | 'éd'<br>sim=0.17629; margin=0.00242; freq=39 |
| step9000 | 256 | 'atom'<br>sim=0.16225; margin=0.000752; freq=15 | 'ervation'<br>sim=0.16488; margin=8.09e-05; freq=8 | 'atom'<br>sim=0.16309; margin=0.000758; freq=15 | 'atom'<br>sim=0.16326; margin=0.000572; freq=15 |
| step9000 | 512 | 'atom'<br>sim=0.16346; margin=0.000449; freq=15 | 'atom'<br>sim=0.16404; margin=0.000313; freq=15 | 'atom'<br>sim=0.16335; margin=0.000521; freq=15 | 'atom'<br>sim=0.16351; margin=0.000458; freq=15 |
| step9000 | 768 | 'atom'<br>sim=0.16362; margin=0.00042; freq=15 | 'atom'<br>sim=0.16367; margin=0.00042; freq=15 | 'atom'<br>sim=0.16353; margin=0.00045; freq=15 | 'atom'<br>sim=0.16359; margin=0.000434; freq=15 |
| step9000 | 1024 | 'atom'<br>sim=0.16361; margin=0.000426; freq=15 | 'atom'<br>sim=0.1636; margin=0.000434; freq=15 | 'atom'<br>sim=0.16359; margin=0.000433; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 |
| step10000 | 256 | 'holder'<br>sim=0.17378; margin=0.0105; freq=3 | 'ervation'<br>sim=0.17812; margin=0.0212; freq=8 | 'ervation'<br>sim=0.17821; margin=0.0175; freq=8 | 'holder'<br>sim=0.1742; margin=0.0123; freq=3 |
| step10000 | 512 | 'ervation'<br>sim=0.17555; margin=0.0177; freq=8 | ' NAC'<br>sim=0.16547; margin=0.00389; freq=2 | ' NAC'<br>sim=0.16454; margin=0.00294; freq=2 | 'ervation'<br>sim=0.17795; margin=0.0213; freq=8 |
| step10000 | 768 | 'ervation'<br>sim=0.16823; margin=0.00553; freq=8 | 'holder'<br>sim=0.17461; margin=0.0114; freq=3 | 'holder'<br>sim=0.17222; margin=0.0101; freq=3 | ' NAC'<br>sim=0.16541; margin=0.00363; freq=2 |
| step10000 | 1024 | 'holder'<br>sim=0.17267; margin=0.00841; freq=3 | 'ervation'<br>sim=0.17693; margin=0.0206; freq=8 | 'ervation'<br>sim=0.1785; margin=0.0194; freq=8 | 'holder'<br>sim=0.1746; margin=0.0113; freq=3 |
| step13000 | 256 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 512 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 768 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 1024 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step21000 | 256 | ' undocumented'<br>sim=0.17902; margin=0.00469; freq=0 | ' pathogen'<br>sim=0.18759; margin=0.0092; freq=1 | 'settings'<br>sim=0.18443; margin=0.0205; freq=0 | 'прав'<br>sim=0.17587; margin=0.00328; freq=0 |
| step21000 | 512 | ' gö'<br>sim=0.16073; margin=0.00249; freq=0 | '性'<br>sim=0.17222; margin=0.00111; freq=0 | 'struct'<br>sim=0.17269; margin=0.00578; freq=1 | '�'<br>sim=0.18293; margin=0.00608; freq=7 |
| step21000 | 768 | 'прав'<br>sim=0.20236; margin=0.0259; freq=0 | ' overex'<br>sim=0.17443; margin=1.68e-05; freq=2 | ' shack'<br>sim=0.20132; margin=0.036; freq=3 | '情'<br>sim=0.16092; margin=0.00437; freq=0 |
| step21000 | 1024 | '性'<br>sim=0.19212; margin=0.0173; freq=0 | ' intimate'<br>sim=0.18122; margin=0.00445; freq=20 | ' sinus'<br>sim=0.16902; margin=0.00252; freq=0 | ' event'<br>sim=0.17662; margin=0.0112; freq=285 |
| step25000 | 256 | ' modul'<br>sim=0.18017; margin=0.00105; freq=0 | ' candidate'<br>sim=0.1656; margin=0.00454; freq=69 | '基'<br>sim=0.19105; margin=0.0138; freq=0 | ' mur'<br>sim=0.17696; margin=0.0161; freq=12 |
| step25000 | 512 | 'network'<br>sim=0.16358; margin=0.00291; freq=0 | 'registry'<br>sim=0.16874; margin=0.000256; freq=0 | ' ST'<br>sim=0.17507; margin=0.00129; freq=27 | ' modul'<br>sim=0.16913; margin=0.00556; freq=0 |
| step25000 | 768 | 'network'<br>sim=0.17175; margin=0.00465; freq=0 | ' profile'<br>sim=0.17842; margin=0.0116; freq=52 | '244'<br>sim=0.17684; margin=0.0047; freq=1 | ' ST'<br>sim=0.17628; margin=0.00137; freq=27 |
| step25000 | 1024 | ' ST'<br>sim=0.18178; margin=0.0175; freq=27 | '基'<br>sim=0.19269; margin=0.0149; freq=0 | ' profile'<br>sim=0.17734; margin=0.00863; freq=52 | ' ST'<br>sim=0.16735; margin=0.004; freq=27 |
| step29000 | 256 | 'eth'<br>sim=0.16153; margin=0.000401; freq=95 | ' committed'<br>sim=0.17122; margin=0.00346; freq=64 | 'eth'<br>sim=0.16171; margin=0.000287; freq=95 | ' committed'<br>sim=0.17226; margin=0.0016; freq=64 |
| step29000 | 512 | 'aptic'<br>sim=0.18538; margin=0.00782; freq=0 | ' committed'<br>sim=0.19072; margin=0.0153; freq=64 | 'aptic'<br>sim=0.18559; margin=0.00757; freq=0 | ' committed'<br>sim=0.19291; margin=0.0179; freq=64 |
| step29000 | 768 | ' committed'<br>sim=0.19789; margin=0.0227; freq=64 | ' committed'<br>sim=0.17291; margin=0.00238; freq=64 | ' committed'<br>sim=0.19696; margin=0.0215; freq=64 | ' committed'<br>sim=0.1696; margin=0.000808; freq=64 |
| step29000 | 1024 | 'undefined'<br>sim=0.15832; margin=0.000378; freq=0 | ' committed'<br>sim=0.16684; margin=0.00486; freq=64 | 'eth'<br>sim=0.15888; margin=5.41e-05; freq=95 | ' committed'<br>sim=0.16838; margin=0.00618; freq=64 |
| step33000 | 256 | 'με'<br>sim=0.17364; margin=0.00628; freq=1 | ' taxonomic'<br>sim=0.18941; margin=0.0127; freq=11 | 'με'<br>sim=0.20227; margin=0.0251; freq=1 | ' acquainted'<br>sim=0.18371; margin=0.0049; freq=5 |
| step33000 | 512 | ' ide'<br>sim=0.1883; margin=0.0101; freq=3 | 'ou'<br>sim=0.18893; margin=0.0187; freq=319 | '196'<br>sim=0.18642; margin=0.00629; freq=0 | 'με'<br>sim=0.19986; margin=0.0223; freq=1 |
| step33000 | 768 | ' sites'<br>sim=0.1692; margin=0.000541; freq=124 | 'με'<br>sim=0.18867; margin=0.0105; freq=1 | ' ba'<br>sim=0.17466; margin=0.00609; freq=22 | 'Е'<br>sim=0.17402; margin=0.00971; freq=0 |
| step33000 | 1024 | ' ba'<br>sim=0.16949; margin=0.00559; freq=22 | '2222'<br>sim=0.18368; margin=0.0102; freq=0 | ' ba'<br>sim=0.18773; margin=0.0127; freq=22 | ' Kant'<br>sim=0.17254; margin=0.00994; freq=16 |
| step37000 | 256 | ' NS'<br>sim=0.16948; margin=0.00873; freq=54 | ' NS'<br>sim=0.18151; margin=0.0014; freq=54 | ' covariance'<br>sim=0.15661; margin=0.00298; freq=0 | 'textsf'<br>sim=0.19347; margin=0.0327; freq=0 |
| step37000 | 512 | ' NS'<br>sim=0.17712; margin=0.00427; freq=54 | 'textsf'<br>sim=0.17679; margin=0.00909; freq=0 | 'textsf'<br>sim=0.17675; margin=0.00227; freq=0 | 'textsf'<br>sim=0.18555; margin=0.00662; freq=0 |
| step37000 | 768 | 'textsf'<br>sim=0.17035; margin=0.0146; freq=0 | 'mes'<br>sim=0.17933; margin=0.00028; freq=35 | 'mes'<br>sim=0.17923; margin=0.00337; freq=35 | 'textsf'<br>sim=0.16584; margin=0.00909; freq=0 |
| step37000 | 1024 | 'textsf'<br>sim=0.17346; margin=0.00408; freq=0 | 'textsf'<br>sim=0.17412; margin=0.00602; freq=0 | ' NS'<br>sim=0.17759; margin=0.000346; freq=54 | ' NS'<br>sim=0.18039; margin=0.00549; freq=54 |
| step41000 | 256 | 'uring'<br>sim=0.1674; margin=0.00773; freq=67 | ' digitally'<br>sim=0.16563; margin=0.0043; freq=16 | ' DR'<br>sim=0.18992; margin=0.0204; freq=7 | ' intracranial'<br>sim=0.16868; margin=1.22e-05; freq=0 |
| step41000 | 512 | ' DR'<br>sim=0.18878; margin=0.0221; freq=7 | ' borderline'<br>sim=0.18229; margin=0.00719; freq=3 | 'uring'<br>sim=0.16421; margin=0.00361; freq=67 | 'window'<br>sim=0.18129; margin=0.0186; freq=0 |
| step41000 | 768 | 'uring'<br>sim=0.16693; margin=0.00772; freq=67 | ' digitally'<br>sim=0.16278; margin=0.00402; freq=16 | ' DR'<br>sim=0.1901; margin=0.0216; freq=7 | ' digitally'<br>sim=0.17327; margin=0.00243; freq=16 |
| step41000 | 1024 | ' DR'<br>sim=0.18737; margin=0.0196; freq=7 | ' borderline'<br>sim=0.18149; margin=0.00804; freq=3 | 'uring'<br>sim=0.16569; margin=0.00548; freq=67 | 'window'<br>sim=0.18213; margin=0.0202; freq=0 |
| step53000 | 256 | 'widget'<br>sim=0.18989; margin=0.00932; freq=0 | 'SHORT'<br>sim=0.19359; margin=0.00624; freq=0 | 'widget'<br>sim=0.18421; margin=0.000815; freq=0 | 'wx'<br>sim=0.17279; margin=0.00194; freq=0 |
| step53000 | 512 | 'ptic'<br>sim=0.17622; margin=0.0017; freq=7 | ' han'<br>sim=0.16606; margin=0.00402; freq=1 | 'ptic'<br>sim=0.1823; margin=0.00413; freq=7 | 'widget'<br>sim=0.18686; margin=0.00492; freq=0 |
| step53000 | 768 | '                                     '<br>sim=0.18797; margin=0.0189; freq=0 | 'widget'<br>sim=0.18886; margin=0.00795; freq=0 | '                                     '<br>sim=0.18159; margin=0.00202; freq=0 | 'oker'<br>sim=0.18981; margin=0.00993; freq=16 |
| step53000 | 1024 | 'oker'<br>sim=0.20177; margin=0.00339; freq=16 | 'ptic'<br>sim=0.17751; margin=0.00615; freq=7 | 'oker'<br>sim=0.21341; margin=0.0311; freq=16 | '                                     '<br>sim=0.18695; margin=0.00645; freq=0 |
| step57000 | 256 | 'effects'<br>sim=0.17849; margin=0.000333; freq=0 | 'Layout'<br>sim=0.18391; margin=0.00764; freq=0 | 'Layout'<br>sim=0.18079; margin=0.00389; freq=0 | ' GC'<br>sim=0.17294; margin=0.00313; freq=14 |
| step57000 | 512 | '$}'<br>sim=0.19703; margin=0.00853; freq=0 | '--------------------------------------------------------------------------------------------------------------------------------'<br>sim=0.17897; margin=7.23e-05; freq=0 | 'izable'<br>sim=0.18089; margin=0.00139; freq=7 | 'izable'<br>sim=0.18098; margin=0.00968; freq=7 |
| step57000 | 768 | 'Layout'<br>sim=0.18462; margin=0.00663; freq=0 | 'context'<br>sim=0.17553; margin=0.00188; freq=0 | 'context'<br>sim=0.17267; margin=7.24e-05; freq=0 | ' GC'<br>sim=0.17997; margin=0.00388; freq=14 |
| step57000 | 1024 | ' movement'<br>sim=0.17698; margin=0.00319; freq=175 | 'izable'<br>sim=0.18107; margin=0.00482; freq=7 | 'izable'<br>sim=0.1808; margin=0.00647; freq=7 | 'izable'<br>sim=0.17971; margin=0.0118; freq=7 |
| step61000 | 256 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 512 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 768 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 1024 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |

### 5.2 Euclidean最近词

| checkpoint | dynamic step | ' clones' | ' motive' | ' cabinet' | ' miles' |
|---|---:|---|---|---|---|
| step5000 | 256 | 'GAA'<br>d=37.295; rel-margin=0.000483; freq=0 | 'GAA'<br>d=37.318; rel-margin=0.000845; freq=0 | 'GAA'<br>d=37.211; rel-margin=0.0008; freq=0 | 'GAA'<br>d=37.472; rel-margin=0.000184; freq=0 |
| step5000 | 512 | 'GAA'<br>d=37.287; rel-margin=0.000654; freq=0 | 'GAA'<br>d=37.5; rel-margin=0.000325; freq=0 | ' maintaining'<br>d=37.107; rel-margin=1.06e-05; freq=42 | ' maintaining'<br>d=37.599; rel-margin=0.000548; freq=42 |
| step5000 | 768 | 'GAA'<br>d=37.432; rel-margin=0.000455; freq=0 | 'GAA'<br>d=37.576; rel-margin=4.67e-05; freq=0 | 'GAA'<br>d=37.157; rel-margin=0.0008; freq=0 | 'GAA'<br>d=37.421; rel-margin=0.000235; freq=0 |
| step5000 | 1024 | 'GAA'<br>d=37.332; rel-margin=0.000867; freq=0 | 'GAA'<br>d=37.257; rel-margin=0.000993; freq=0 | 'GAA'<br>d=37.511; rel-margin=2.21e-05; freq=0 | 'GAA'<br>d=37.164; rel-margin=0.000497; freq=0 |
| step7000 | 256 | ' preserv'<br>d=38.332; rel-margin=0.00049; freq=2 | 'ynamic'<br>d=39.839; rel-margin=0.000527; freq=1 | ' repertoire'<br>d=37.769; rel-margin=6.44e-05; freq=27 | ' politique'<br>d=40.025; rel-margin=0.000197; freq=0 |
| step7000 | 512 | 'hospital'<br>d=38.285; rel-margin=0.00051; freq=3 | 'hospital'<br>d=38.415; rel-margin=8.32e-05; freq=3 | 'コ'<br>d=38.643; rel-margin=0.000291; freq=2 | ' meaningless'<br>d=37.109; rel-margin=3.43e-05; freq=8 |
| step7000 | 768 | ' politique'<br>d=39.415; rel-margin=0.000599; freq=0 | 'hospital'<br>d=38.168; rel-margin=0.000265; freq=3 | 'コ'<br>d=38.473; rel-margin=0.000394; freq=2 | ' politique'<br>d=39.72; rel-margin=0.000521; freq=0 |
| step7000 | 1024 | 'avirus'<br>d=40.38; rel-margin=0.000454; freq=2 | ' prevalent'<br>d=40.217; rel-margin=0.000124; freq=20 | 'コ'<br>d=38.449; rel-margin=0.0004; freq=2 | 'Tour'<br>d=38.441; rel-margin=5.64e-05; freq=1 |
| step8000 | 256 | '性'<br>d=37.988; rel-margin=0.000142; freq=0 | 'Tour'<br>d=38.244; rel-margin=0.000364; freq=1 | ' retrieving'<br>d=39.771; rel-margin=2.65e-05; freq=1 | 'Ip'<br>d=41.213; rel-margin=7.92e-05; freq=0 |
| step8000 | 512 | ' CBC'<br>d=37.806; rel-margin=2.56e-05; freq=2 | 'Tour'<br>d=38.06; rel-margin=6.85e-05; freq=1 | 'Tour'<br>d=39.346; rel-margin=0.00101; freq=1 | ' habit'<br>d=38.414; rel-margin=0.000321; freq=26 |
| step8000 | 768 | '性'<br>d=38.379; rel-margin=1.7e-05; freq=0 | 'edom'<br>d=37.932; rel-margin=0.000211; freq=0 | 'edom'<br>d=37.868; rel-margin=0.000198; freq=0 | ' libert'<br>d=37.145; rel-margin=0.00113; freq=2 |
| step8000 | 1024 | ' retrieving'<br>d=39.88; rel-margin=7.94e-06; freq=1 | '性'<br>d=38.196; rel-margin=0.000135; freq=0 | ' SUR'<br>d=38.15; rel-margin=4.38e-05; freq=1 | 'éd'<br>d=39.684; rel-margin=0.000231; freq=39 |
| step9000 | 256 | 'ervation'<br>d=38.454; rel-margin=5.15e-05; freq=8 | 'ervation'<br>d=38.752; rel-margin=8.36e-05; freq=8 | 'ervation'<br>d=38.613; rel-margin=5.17e-05; freq=8 | 'ervation'<br>d=38.583; rel-margin=5.87e-05; freq=8 |
| step9000 | 512 | 'ervation'<br>d=38.591; rel-margin=6.34e-05; freq=8 | 'ervation'<br>d=38.675; rel-margin=6.87e-05; freq=8 | 'ervation'<br>d=38.591; rel-margin=6.07e-05; freq=8 | 'ervation'<br>d=38.604; rel-margin=6.3e-05; freq=8 |
| step9000 | 768 | 'ervation'<br>d=38.616; rel-margin=6.45e-05; freq=8 | 'ervation'<br>d=38.628; rel-margin=6.45e-05; freq=8 | 'ervation'<br>d=38.608; rel-margin=6.33e-05; freq=8 | 'ervation'<br>d=38.614; rel-margin=6.39e-05; freq=8 |
| step9000 | 1024 | 'ervation'<br>d=38.617; rel-margin=6.43e-05; freq=8 | 'ervation'<br>d=38.617; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.614; rel-margin=6.4e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 |
| step10000 | 256 | 'holder'<br>d=40.07; rel-margin=0.000462; freq=3 | 'ervation'<br>d=39.072; rel-margin=0.000792; freq=8 | 'ervation'<br>d=38.953; rel-margin=0.000661; freq=8 | 'holder'<br>d=39.94; rel-margin=0.00053; freq=3 |
| step10000 | 512 | 'ervation'<br>d=39.246; rel-margin=0.000579; freq=8 | 'コ'<br>d=39.429; rel-margin=6.1e-06; freq=2 | 'holder'<br>d=39.783; rel-margin=7.46e-05; freq=3 | 'ervation'<br>d=39.084; rel-margin=0.000776; freq=8 |
| step10000 | 768 | 'ervation'<br>d=39.147; rel-margin=0.000211; freq=8 | 'holder'<br>d=40.026; rel-margin=0.000496; freq=3 | 'holder'<br>d=39.813; rel-margin=0.000452; freq=3 | 'ervation'<br>d=39.413; rel-margin=4.84e-06; freq=8 |
| step10000 | 1024 | 'holder'<br>d=40.129; rel-margin=0.000386; freq=3 | 'ervation'<br>d=39.162; rel-margin=0.00069; freq=8 | 'ervation'<br>d=39.003; rel-margin=0.000739; freq=8 | 'holder'<br>d=40.034; rel-margin=0.000492; freq=3 |
| step13000 | 256 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 512 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 768 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 1024 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step21000 | 256 | ' undocumented'<br>d=46.117; rel-margin=0.000173; freq=0 | ' pathogen'<br>d=46.823; rel-margin=0.000414; freq=1 | 'settings'<br>d=46.473; rel-margin=0.000716; freq=0 | 'прав'<br>d=46.245; rel-margin=0.000142; freq=0 |
| step21000 | 512 | ' gö'<br>d=45.606; rel-margin=4.77e-05; freq=0 | '性'<br>d=46.244; rel-margin=2.77e-05; freq=0 | ' descriptor'<br>d=46.929; rel-margin=6.24e-05; freq=1 | '�'<br>d=45.68; rel-margin=0.000102; freq=7 |
| step21000 | 768 | 'прав'<br>d=47.878; rel-margin=0.000864; freq=0 | ' overex'<br>d=46.347; rel-margin=1.91e-05; freq=2 | ' shack'<br>d=47.801; rel-margin=0.00118; freq=3 | '情'<br>d=49.037; rel-margin=0.000161; freq=0 |
| step21000 | 1024 | '性'<br>d=45.405; rel-margin=0.00055; freq=0 | ' intimate'<br>d=47.276; rel-margin=0.000259; freq=20 | ' sinus'<br>d=46.886; rel-margin=0.000186; freq=0 | ' event'<br>d=44.061; rel-margin=0.000269; freq=285 |
| step25000 | 256 | ' selective'<br>d=49.124; rel-margin=2.77e-05; freq=14 | ' candidate'<br>d=49.482; rel-margin=0.000138; freq=69 | '基'<br>d=49.885; rel-margin=0.000422; freq=0 | ' mur'<br>d=48.962; rel-margin=0.000523; freq=12 |
| step25000 | 512 | 'network'<br>d=50.44; rel-margin=8.06e-05; freq=0 | '基'<br>d=49.17; rel-margin=2.79e-05; freq=0 | ' ST'<br>d=50.11; rel-margin=5.51e-05; freq=27 | ' modul'<br>d=48.611; rel-margin=0.00015; freq=0 |
| step25000 | 768 | 'network'<br>d=50.46; rel-margin=0.000119; freq=0 | ' profile'<br>d=50.974; rel-margin=0.000176; freq=52 | '244'<br>d=48.964; rel-margin=0.000199; freq=1 | ' ST'<br>d=49.978; rel-margin=5.78e-05; freq=27 |
| step25000 | 1024 | ' ST'<br>d=50.511; rel-margin=0.000561; freq=27 | '基'<br>d=49.898; rel-margin=0.000455; freq=0 | ' profile'<br>d=50.867; rel-margin=0.000204; freq=52 | ' ST'<br>d=50.649; rel-margin=0.000119; freq=27 |
| step29000 | 256 | 'eth'<br>d=51.071; rel-margin=8.07e-05; freq=95 | ' committed'<br>d=51.129; rel-margin=0.000125; freq=64 | 'eth'<br>d=51.073; rel-margin=8.09e-05; freq=95 | ' committed'<br>d=51.132; rel-margin=7.06e-05; freq=64 |
| step29000 | 512 | 'aptic'<br>d=51.115; rel-margin=0.000203; freq=0 | ' committed'<br>d=51.243; rel-margin=0.000451; freq=64 | 'aptic'<br>d=51.122; rel-margin=0.000195; freq=0 | ' committed'<br>d=51.246; rel-margin=0.000507; freq=64 |
| step29000 | 768 | ' committed'<br>d=51.326; rel-margin=0.000808; freq=64 | ' committed'<br>d=51.347; rel-margin=0.000483; freq=64 | ' committed'<br>d=51.345; rel-margin=0.000771; freq=64 | ' committed'<br>d=51.307; rel-margin=0.000431; freq=64 |
| step29000 | 1024 | 'eth'<br>d=51.079; rel-margin=5.09e-05; freq=95 | ' committed'<br>d=51.11; rel-margin=8.5e-05; freq=64 | 'eth'<br>d=51.074; rel-margin=5.91e-05; freq=95 | ' committed'<br>d=51.117; rel-margin=9.86e-05; freq=64 |
| step33000 | 256 | 'με'<br>d=54.093; rel-margin=3.66e-05; freq=1 | ' taxonomic'<br>d=55.263; rel-margin=0.00035; freq=11 | 'με'<br>d=53.659; rel-margin=0.000592; freq=1 | ' acquainted'<br>d=51.885; rel-margin=9.34e-05; freq=5 |
| step33000 | 512 | ' ide'<br>d=55.088; rel-margin=0.000393; freq=3 | 'ou'<br>d=55.473; rel-margin=0.000545; freq=319 | '196'<br>d=52.216; rel-margin=4.65e-05; freq=0 | 'με'<br>d=54.686; rel-margin=0.00051; freq=1 |
| step33000 | 768 | ' event'<br>d=52.07; rel-margin=2.29e-05; freq=285 | 'με'<br>d=54.602; rel-margin=0.000377; freq=1 | ' ba'<br>d=54.345; rel-margin=0.000137; freq=22 | 'Е'<br>d=52.753; rel-margin=0.000254; freq=0 |
| step33000 | 1024 | ' ba'<br>d=53.21; rel-margin=0.000151; freq=22 | '2222'<br>d=54.441; rel-margin=0.000136; freq=0 | ' ba'<br>d=55.162; rel-margin=0.000336; freq=22 | ' Kant'<br>d=53.047; rel-margin=0.000212; freq=16 |
| step37000 | 256 | ' NS'<br>d=54.428; rel-margin=0.000291; freq=54 | ' NS'<br>d=52.697; rel-margin=0.000229; freq=54 | ' covariance'<br>d=53.026; rel-margin=2.88e-05; freq=0 | 'textsf'<br>d=52.119; rel-margin=0.000658; freq=0 |
| step37000 | 512 | ' NS'<br>d=55.114; rel-margin=0.000173; freq=54 | 'fr'<br>d=55.628; rel-margin=7.95e-06; freq=36 | ' NS'<br>d=55.164; rel-margin=0.000123; freq=54 | ' NS'<br>d=53.841; rel-margin=8.07e-05; freq=54 |
| step37000 | 768 | 'textsf'<br>d=52.134; rel-margin=0.000163; freq=0 | 'mes'<br>d=50.722; rel-margin=0.000363; freq=35 | 'mes'<br>d=50.539; rel-margin=0.000447; freq=35 | 'comments'<br>d=55.401; rel-margin=1.18e-05; freq=0 |
| step37000 | 1024 | 'fr'<br>d=55.514; rel-margin=8.31e-05; freq=36 | ' NS'<br>d=55.758; rel-margin=2.65e-05; freq=54 | ' NS'<br>d=55.139; rel-margin=0.000241; freq=54 | ' NS'<br>d=55.133; rel-margin=0.000289; freq=54 |
| step41000 | 256 | 'uring'<br>d=58.331; rel-margin=0.000167; freq=67 | ' digitally'<br>d=56.458; rel-margin=0.000133; freq=16 | ' DR'<br>d=53.98; rel-margin=0.00054; freq=7 | ' digitally'<br>d=55.584; rel-margin=6.25e-06; freq=16 |
| step41000 | 512 | ' DR'<br>d=53.413; rel-margin=0.000601; freq=7 | ' borderline'<br>d=55.362; rel-margin=0.000203; freq=3 | 'uring'<br>d=58.126; rel-margin=6.69e-05; freq=67 | 'window'<br>d=56.231; rel-margin=0.000424; freq=0 |
| step41000 | 768 | 'uring'<br>d=58.417; rel-margin=0.000167; freq=67 | ' digitally'<br>d=56.507; rel-margin=0.000124; freq=16 | ' DR'<br>d=53.752; rel-margin=0.000576; freq=7 | ' digitally'<br>d=55.64; rel-margin=6.99e-05; freq=16 |
| step41000 | 1024 | ' DR'<br>d=53.254; rel-margin=0.000598; freq=7 | ' borderline'<br>d=55.309; rel-margin=0.000226; freq=3 | 'uring'<br>d=58.235; rel-margin=0.000113; freq=67 | 'window'<br>d=56.103; rel-margin=0.000467; freq=0 |
| step53000 | 256 | 'widget'<br>d=58.791; rel-margin=0.000116; freq=0 | 'SHORT'<br>d=59.628; rel-margin=4.08e-05; freq=0 | 'widget'<br>d=59.197; rel-margin=5.48e-05; freq=0 | 'wx'<br>d=56.839; rel-margin=6.47e-05; freq=0 |
| step53000 | 512 | 'ptic'<br>d=56.532; rel-margin=7.36e-06; freq=7 | 'igion'<br>d=61.269; rel-margin=2.71e-05; freq=1 | 'ptic'<br>d=55.262; rel-margin=0.000148; freq=7 | 'ер'<br>d=58.96; rel-margin=1.2e-05; freq=0 |
| step53000 | 768 | 'ات'<br>d=61.154; rel-margin=3.84e-05; freq=0 | 'widget'<br>d=58.824; rel-margin=0.000138; freq=0 | ' Games'<br>d=59.497; rel-margin=1e-05; freq=164 | 'oker'<br>d=56.243; rel-margin=0.000213; freq=16 |
| step53000 | 1024 | 'oker'<br>d=56.316; rel-margin=0.000416; freq=16 | 'ptic'<br>d=56.425; rel-margin=0.000117; freq=7 | 'oker'<br>d=55.536; rel-margin=0.00102; freq=16 | ' committing'<br>d=62.128; rel-margin=0.000199; freq=10 |
| step57000 | 256 | 'effects'<br>d=61.594; rel-margin=3.44e-05; freq=0 | 'Layout'<br>d=61.798; rel-margin=0.000145; freq=0 | 'Layout'<br>d=61.745; rel-margin=9.57e-05; freq=0 | ' GC'<br>d=60.657; rel-margin=1.3e-05; freq=14 |
| step57000 | 512 | '$}'<br>d=79.233; rel-margin=4.43e-05; freq=0 | '--------------------------------------------------------------------------------------------------------------------------------'<br>d=66.824; rel-margin=1.19e-05; freq=0 | 'izable'<br>d=66.483; rel-margin=1.76e-05; freq=7 | 'izable'<br>d=64.752; rel-margin=0.00019; freq=7 |
| step57000 | 768 | 'Layout'<br>d=61.769; rel-margin=0.000123; freq=0 | 'effects'<br>d=61.415; rel-margin=3.6e-06; freq=0 | 'effects'<br>d=61.15; rel-margin=4.25e-05; freq=0 | ' GC'<br>d=59.823; rel-margin=5.77e-05; freq=14 |
| step57000 | 1024 | ' movement'<br>d=68.728; rel-margin=7.85e-05; freq=175 | 'izable'<br>d=65.875; rel-margin=8.77e-05; freq=7 | 'izable'<br>d=65.528; rel-margin=0.000122; freq=7 | 'izable'<br>d=63.584; rel-margin=0.000237; freq=7 |
| step61000 | 256 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 |
| step61000 | 512 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 768 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 1024 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |

### 5.3 LM-head最近词

| checkpoint | dynamic step | ' clones' | ' motive' | ' cabinet' | ' miles' |
|---|---:|---|---|---|---|
| step5000 | 256 | '\n'<br>p=0.0705; Δp=0.0109; Δlogit=0.168; freq=1 | '\n'<br>p=0.0942; Δp=0.0486; Δlogit=0.726; freq=1 | '\n'<br>p=0.08931; Δp=0.0129; Δlogit=0.156; freq=1 | '\n'<br>p=0.07528; Δp=0.0224; Δlogit=0.354; freq=1 |
| step5000 | 512 | '\n'<br>p=0.07584; Δp=0.0137; Δlogit=0.199; freq=1 | '\n'<br>p=0.05975; Δp=0.0106; Δlogit=0.195; freq=1 | ','<br>p=0.05834; Δp=0.00772; Δlogit=0.142; freq=2711 | ','<br>p=0.0564; Δp=0.00168; Δlogit=0.0302; freq=2711 |
| step5000 | 768 | '\n'<br>p=0.06233; Δp=0.0186; Δlogit=0.355; freq=1 | ','<br>p=0.05318; Δp=0.00795; Δlogit=0.162; freq=2711 | '\n'<br>p=0.09198; Δp=0.0233; Δlogit=0.293; freq=1 | ' class'<br>p=0.07534; Δp=0.00745; Δlogit=0.104; freq=348 |
| step5000 | 1024 | '\n'<br>p=0.08865; Δp=0.0274; Δlogit=0.37; freq=1 | '\n'<br>p=0.07571; Δp=0.00917; Δlogit=0.129; freq=1 | '\n'<br>p=0.08459; Δp=0.0386; Δlogit=0.609; freq=1 | ','<br>p=0.06148; Δp=0.0112; Δlogit=0.201; freq=2711 |
| step7000 | 256 | ','<br>p=0.07458; Δp=0.0409; Δlogit=0.796; freq=2711 | '.'<br>p=0.07156; Δp=0.0303; Δlogit=0.551; freq=8666 | ' in'<br>p=0.03639; Δp=0.00714; Δlogit=0.218; freq=39777 | '.'<br>p=0.05065; Δp=0.0191; Δlogit=0.474; freq=8666 |
| step7000 | 512 | '\n'<br>p=0.1717; Δp=0.0898; Δlogit=0.741; freq=1 | '\n'<br>p=0.126; Δp=0.0779; Δlogit=0.964; freq=1 | '.'<br>p=0.04543; Δp=0.0107; Δlogit=0.269; freq=8666 | '\n'<br>p=0.0419; Δp=0.00993; Δlogit=0.27; freq=1 |
| step7000 | 768 | '\n'<br>p=0.08287; Δp=0.0221; Δlogit=0.311; freq=1 | '\n'<br>p=0.1802; Δp=0.0964; Δlogit=0.766; freq=1 | '.'<br>p=0.04114; Δp=0.0109; Δlogit=0.308; freq=8666 | '.'<br>p=0.07917; Δp=0.0382; Δlogit=0.659; freq=8666 |
| step7000 | 1024 | '.'<br>p=0.07027; Δp=0.0519; Δlogit=1.34; freq=8666 | '.'<br>p=0.08837; Δp=0.0662; Δlogit=1.38; freq=8666 | '.'<br>p=0.04113; Δp=0.0107; Δlogit=0.303; freq=8666 | '\n'<br>p=0.1221; Δp=0.0803; Δlogit=1.07; freq=1 |
| step8000 | 256 | 's'<br>p=0.1534; Δp=0.103; Δlogit=1.11; freq=17395 | '\n'<br>p=0.1866; Δp=0.0429; Δlogit=0.261; freq=1 | ','<br>p=0.1368; Δp=0.0631; Δlogit=0.618; freq=2711 | '.'<br>p=0.0791; Δp=0.0181; Δlogit=0.26; freq=8666 |
| step8000 | 512 | '\n'<br>p=0.06055; Δp=0.0207; Δlogit=0.418; freq=1 | ' '<br>p=0.05057; Δp=0.00536; Δlogit=0.112; freq=12180 | 'V'<br>p=0.1001; Δp=0.0482; Δlogit=0.657; freq=205 | '.'<br>p=0.1486; Δp=0.0911; Δlogit=0.949; freq=8666 |
| step8000 | 768 | '\n'<br>p=0.03724; Δp=0.00344; Δlogit=0.097; freq=1 | 'V'<br>p=0.09528; Δp=0.0613; Δlogit=1.03; freq=205 | 'V'<br>p=0.06013; Δp=0.0217; Δlogit=0.448; freq=205 | '\n'<br>p=0.0877; Δp=0.0176; Δlogit=0.223; freq=1 |
| step8000 | 1024 | ','<br>p=0.1368; Δp=0.0648; Δlogit=0.643; freq=2711 | ' '<br>p=0.04757; Δp=0.00442; Δlogit=0.0975; freq=12180 | '\n'<br>p=0.05415; Δp=0.0205; Δlogit=0.477; freq=1 | ' I'<br>p=0.08089; Δp=0.00204; Δlogit=0.0256; freq=2877 |
| step9000 | 256 | '\n'<br>p=0.2185; Δp=0.18; Δlogit=1.74; freq=1 | '\n'<br>p=0.2261; Δp=0.187; Δlogit=1.75; freq=1 | '\n'<br>p=0.2133; Δp=0.173; Δlogit=1.67; freq=1 | '\n'<br>p=0.2178; Δp=0.179; Δlogit=1.72; freq=1 |
| step9000 | 512 | '\n'<br>p=0.2203; Δp=0.182; Δlogit=1.74; freq=1 | '\n'<br>p=0.2201; Δp=0.181; Δlogit=1.72; freq=1 | '\n'<br>p=0.2184; Δp=0.179; Δlogit=1.73; freq=1 | '\n'<br>p=0.2194; Δp=0.181; Δlogit=1.73; freq=1 |
| step9000 | 768 | '\n'<br>p=0.2198; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2193; Δp=0.18; Δlogit=1.73; freq=1 | '\n'<br>p=0.2194; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 |
| step9000 | 1024 | '\n'<br>p=0.2196; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2194; Δp=0.18; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 |
| step10000 | 256 | '\n'<br>p=0.128; Δp=0.0758; Δlogit=0.897; freq=1 | ' '<br>p=0.08399; Δp=0.00317; Δlogit=0.0385; freq=12180 | ' '<br>p=0.08732; Δp=0.000257; Δlogit=0.00295; freq=12180 | '\n'<br>p=0.1141; Δp=0.0696; Δlogit=0.941; freq=1 |
| step10000 | 512 | '\n'<br>p=0.08171; Δp=0.00843; Δlogit=0.109; freq=1 | '\n'<br>p=0.1447; Δp=0.0784; Δlogit=0.781; freq=1 | '\n'<br>p=0.1552; Δp=0.0864; Δlogit=0.815; freq=1 | ' '<br>p=0.08332; Δp=0.00231; Δlogit=0.0281; freq=12180 |
| step10000 | 768 | '\n'<br>p=0.1249; Δp=0.054; Δlogit=0.566; freq=1 | '\n'<br>p=0.1209; Δp=0.0729; Δlogit=0.923; freq=1 | '\n'<br>p=0.1059; Δp=0.0649; Δlogit=0.948; freq=1 | '\n'<br>p=0.1438; Δp=0.0778; Δlogit=0.778; freq=1 |
| step10000 | 1024 | '\n'<br>p=0.1353; Δp=0.0794; Δlogit=0.883; freq=1 | '\n'<br>p=0.08078; Δp=0.00188; Δlogit=0.0235; freq=1 | ' '<br>p=0.08657; Δp=0.00327; Δlogit=0.0385; freq=12180 | '\n'<br>p=0.1216; Δp=0.0732; Δlogit=0.921; freq=1 |
| step13000 | 256 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 512 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 768 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 1024 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step21000 | 256 | ','<br>p=0.0847; Δp=0.0199; Δlogit=0.267; freq=2711 | '.'<br>p=0.09131; Δp=0.0184; Δlogit=0.226; freq=8666 | ' I'<br>p=0.03856; Δp=0.00261; Δlogit=0.0701; freq=2877 | ' me'<br>p=0.1792; Δp=0.0597; Δlogit=0.405; freq=357 |
| step21000 | 512 | '\n'<br>p=0.05707; Δp=0.026; Δlogit=0.608; freq=1 | '\n'<br>p=0.07815; Δp=0.0215; Δlogit=0.322; freq=1 | ' ('<br>p=0.05468; Δp=0.00108; Δlogit=0.0199; freq=11992 | ' med'<br>p=0.03339; Δp=0.000404; Δlogit=0.0122; freq=37 |
| step21000 | 768 | ' '<br>p=0.04885; Δp=0.0145; Δlogit=0.353; freq=12180 | ','<br>p=0.1124; Δp=0.0102; Δlogit=0.0952; freq=2711 | '.'<br>p=0.1626; Δp=0.0274; Δlogit=0.185; freq=8666 | 'us'<br>p=0.04601; Δp=0.0211; Δlogit=0.613; freq=1269 |
| step21000 | 1024 | '\n'<br>p=0.09141; Δp=0.0194; Δlogit=0.239; freq=1 | '.'<br>p=0.09834; Δp=0.0113; Δlogit=0.122; freq=8666 | '\n'<br>p=0.102; Δp=0.0547; Δlogit=0.769; freq=1 | '\n'<br>p=0.04263; Δp=0.00305; Δlogit=0.0743; freq=1 |
| step25000 | 256 | '\n'<br>p=0.1148; Δp=0.0741; Δlogit=1.04; freq=1 | '~'<br>p=0.2925; Δp=0.259; Δlogit=2.17; freq=0 | '\n'<br>p=0.09066; Δp=0.0651; Δlogit=1.27; freq=1 | '~'<br>p=0.1413; Δp=0.108; Δlogit=1.44; freq=0 |
| step25000 | 512 | '\n'<br>p=0.1904; Δp=0.144; Δlogit=1.4; freq=1 | ' and'<br>p=0.1023; Δp=0.0586; Δlogit=0.851; freq=50606 | '\n'<br>p=0.301; Δp=0.259; Δlogit=1.96; freq=1 | '.'<br>p=0.04599; Δp=0.00844; Δlogit=0.203; freq=8666 |
| step25000 | 768 | '\n'<br>p=0.2224; Δp=0.164; Δlogit=1.34; freq=1 | '\n'<br>p=0.09577; Δp=0.0228; Δlogit=0.272; freq=1 | '\n'<br>p=0.1509; Δp=0.116; Δlogit=1.46; freq=1 | '\n'<br>p=0.3046; Δp=0.266; Δlogit=2.06; freq=1 |
| step25000 | 1024 | '\n'<br>p=0.114; Δp=0.0696; Δlogit=0.943; freq=1 | '\n'<br>p=0.08349; Δp=0.0566; Δlogit=1.13; freq=1 | '\n'<br>p=0.09812; Δp=0.0283; Δlogit=0.34; freq=1 | '\n'<br>p=0.05309; Δp=0.0301; Δlogit=0.835; freq=1 |
| step29000 | 256 | ' I'<br>p=0.07571; Δp=0.0262; Δlogit=0.424; freq=2877 | ')'<br>p=0.05303; Δp=0.0185; Δlogit=0.43; freq=0 | ' I'<br>p=0.07215; Δp=0.0211; Δlogit=0.346; freq=2877 | ')'<br>p=0.05018; Δp=0.015; Δlogit=0.354; freq=0 |
| step29000 | 512 | ' STATES'<br>p=0.05992; Δp=0.0174; Δlogit=0.343; freq=0 | ' '<br>p=0.05391; Δp=0.0077; Δlogit=0.154; freq=12180 | ' STATES'<br>p=0.05952; Δp=0.0155; Δlogit=0.301; freq=0 | ' '<br>p=0.05583; Δp=0.0125; Δlogit=0.254; freq=12180 |
| step29000 | 768 | ' I'<br>p=0.06695; Δp=0.0276; Δlogit=0.532; freq=2877 | ' I'<br>p=0.09939; Δp=0.0504; Δlogit=0.708; freq=2877 | ' I'<br>p=0.07055; Δp=0.0331; Δlogit=0.632; freq=2877 | ' I'<br>p=0.1016; Δp=0.0507; Δlogit=0.691; freq=2877 |
| step29000 | 1024 | ' I'<br>p=0.09513; Δp=0.0472; Δlogit=0.685; freq=2877 | ')'<br>p=0.05746; Δp=0.0118; Δlogit=0.229; freq=0 | ' I'<br>p=0.0923; Δp=0.045; Δlogit=0.669; freq=2877 | ')'<br>p=0.05694; Δp=0.016; Δlogit=0.329; freq=0 |
| step33000 | 256 | 'A'<br>p=0.1355; Δp=0.0643; Δlogit=0.644; freq=289 | '’'<br>p=0.2298; Δp=0.174; Δlogit=1.42; freq=0 | '~'<br>p=0.08377; Δp=0.0158; Δlogit=0.21; freq=0 | ' A'<br>p=0.08162; Δp=0.00816; Δlogit=0.105; freq=3267 |
| step33000 | 512 | ' DE'<br>p=0.05922; Δp=0.012; Δlogit=0.227; freq=32 | ' STATES'<br>p=0.09189; Δp=0.0392; Δlogit=0.555; freq=0 | ' a'<br>p=0.103; Δp=0.0424; Δlogit=0.53; freq=34407 | ','<br>p=0.0817; Δp=0.00503; Δlogit=0.0635; freq=2711 |
| step33000 | 768 | ' '<br>p=0.1344; Δp=0.0279; Δlogit=0.233; freq=12180 | '\xa0'<br>p=0.06957; Δp=0.00324; Δlogit=0.0478; freq=0 | '’'<br>p=0.1034; Δp=0.0294; Δlogit=0.334; freq=0 | ')'<br>p=0.1059; Δp=0.0114; Δlogit=0.114; freq=0 |
| step33000 | 1024 | '\n'<br>p=0.1237; Δp=0.0857; Δlogit=1.18; freq=1 | 'A'<br>p=0.1086; Δp=0.00909; Δlogit=0.0874; freq=289 | '’'<br>p=0.08796; Δp=0.0214; Δlogit=0.278; freq=0 | '’'<br>p=0.07061; Δp=0.0116; Δlogit=0.179; freq=0 |
| step37000 | 256 | '\xa0'<br>p=0.1195; Δp=0.00916; Δlogit=0.0797; freq=0 | '['<br>p=0.0804; Δp=0.039; Δlogit=0.664; freq=0 | '\xa0'<br>p=0.1193; Δp=0.0627; Δlogit=0.745; freq=0 | 'ubotu'<br>p=0.07444; Δp=0.0125; Δlogit=0.184; freq=0 |
| step37000 | 512 | '\xa0'<br>p=0.144; Δp=0.0655; Δlogit=0.607; freq=0 | '\xa0'<br>p=0.2154; Δp=0.142; Δlogit=1.08; freq=0 | '\xa0'<br>p=0.1305; Δp=0.031; Δlogit=0.271; freq=0 | '['<br>p=0.123; Δp=0.0702; Δlogit=0.846; freq=0 |
| step37000 | 768 | '                        '<br>p=0.108; Δp=0.053; Δlogit=0.674; freq=0 | 'ubotu'<br>p=0.05922; Δp=0.0189; Δlogit=0.385; freq=0 | 'ubotu'<br>p=0.04582; Δp=0.00149; Δlogit=0.0331; freq=0 | '\xa0'<br>p=0.2621; Δp=0.17; Δlogit=1.05; freq=0 |
| step37000 | 1024 | '\xa0'<br>p=0.1944; Δp=0.117; Δlogit=0.927; freq=0 | '\xa0'<br>p=0.1202; Δp=0.0076; Δlogit=0.0653; freq=0 | '['<br>p=0.1306; Δp=0.0473; Δlogit=0.45; freq=0 | '['<br>p=0.1222; Δp=0.0434; Δlogit=0.438; freq=0 |
| step41000 | 256 | ' I'<br>p=0.0915; Δp=0.0145; Δlogit=0.173; freq=2877 | ' I'<br>p=0.07665; Δp=0.0216; Δlogit=0.332; freq=2877 | 'oid'<br>p=0.06885; Δp=0.0224; Δlogit=0.394; freq=65 | ' '<br>p=0.08978; Δp=0.0174; Δlogit=0.215; freq=12180 |
| step41000 | 512 | 'oid'<br>p=0.07745; Δp=0.0242; Δlogit=0.374; freq=65 | ' type'<br>p=0.08615; Δp=0.0571; Δlogit=1.09; freq=241 | ' I'<br>p=0.09338; Δp=0.0177; Δlogit=0.21; freq=2877 | 'sf'<br>p=0.06518; Δp=0.0395; Δlogit=0.93; freq=2 |
| step41000 | 768 | ' I'<br>p=0.08844; Δp=0.0152; Δlogit=0.188; freq=2877 | ' I'<br>p=0.07734; Δp=0.0163; Δlogit=0.236; freq=2877 | 'oid'<br>p=0.0727; Δp=0.0226; Δlogit=0.372; freq=65 | ' '<br>p=0.0771; Δp=0.00485; Δlogit=0.065; freq=12180 |
| step41000 | 1024 | 'oid'<br>p=0.07893; Δp=0.025; Δlogit=0.381; freq=65 | ' type'<br>p=0.08294; Δp=0.055; Δlogit=1.09; freq=241 | ' I'<br>p=0.0924; Δp=0.0158; Δlogit=0.187; freq=2877 | 'sf'<br>p=0.0662; Δp=0.0418; Δlogit=0.997; freq=2 |
| step53000 | 256 | '\xa0'<br>p=0.08908; Δp=0.0242; Δlogit=0.318; freq=0 | '\xa0'<br>p=0.07764; Δp=0.0415; Δlogit=0.766; freq=0 | '\xa0'<br>p=0.06742; Δp=0.00862; Δlogit=0.137; freq=0 | '\n'<br>p=0.05615; Δp=0.0113; Δlogit=0.224; freq=1 |
| step53000 | 512 | '('<br>p=0.06723; Δp=0.0266; Δlogit=0.504; freq=0 | '\n'<br>p=0.09099; Δp=0.015; Δlogit=0.181; freq=1 | '’'<br>p=0.05397; Δp=0.00346; Δlogit=0.0663; freq=0 | '\xa0'<br>p=0.08475; Δp=0.0228; Δlogit=0.314; freq=0 |
| step53000 | 768 | ' '<br>p=0.1185; Δp=0.046; Δlogit=0.491; freq=12180 | '\xa0'<br>p=0.08841; Δp=0.0233; Δlogit=0.306; freq=0 | ' '<br>p=0.0951; Δp=0.02; Δlogit=0.237; freq=12180 | '('<br>p=0.0896; Δp=0.037; Δlogit=0.532; freq=0 |
| step53000 | 1024 | '\n'<br>p=0.06617; Δp=0.00485; Δlogit=0.0762; freq=1 | '('<br>p=0.05808; Δp=0.0149; Δlogit=0.297; freq=0 | '\n'<br>p=0.07571; Δp=0.00638; Δlogit=0.088; freq=1 | ' '<br>p=0.1562; Δp=0.0957; Δlogit=0.948; freq=12180 |
| step57000 | 256 | '^'<br>p=0.03805; Δp=0.00501; Δlogit=0.141; freq=0 | '^'<br>p=0.05748; Δp=0.0163; Δlogit=0.332; freq=0 | '^'<br>p=0.0643; Δp=0.0196; Δlogit=0.363; freq=0 | '**'<br>p=0.09047; Δp=0.00619; Δlogit=0.0709; freq=0 |
| step57000 | 512 | ' BY'<br>p=0.6324; Δp=0.605; Δlogit=3.15; freq=4 | ' Getty'<br>p=0.07863; Δp=0.0175; Δlogit=0.252; freq=2 | ' Getty'<br>p=0.07095; Δp=0.00203; Δlogit=0.029; freq=2 | ' God'<br>p=0.08224; Δp=0.0446; Δlogit=0.782; freq=318 |
| step57000 | 768 | '^'<br>p=0.04695; Δp=0.0118; Δlogit=0.288; freq=0 | '^'<br>p=0.07989; Δp=0.0194; Δlogit=0.278; freq=0 | '^'<br>p=0.08429; Δp=0.0137; Δlogit=0.178; freq=0 | '**'<br>p=0.09356; Δp=0.0338; Δlogit=0.448; freq=0 |
| step57000 | 1024 | ' AP'<br>p=0.05099; Δp=0.00991; Δlogit=0.216; freq=53 | ' God'<br>p=0.08353; Δp=0.0382; Δlogit=0.612; freq=318 | ' God'<br>p=0.08586; Δp=0.0487; Δlogit=0.837; freq=318 | ' God'<br>p=0.06067; Δp=0.0263; Δlogit=0.568; freq=318 |
| step61000 | 256 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 512 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 768 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 1024 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |

完整机器可读结果见[`window_endpoint_metrics.csv`](processed/window_endpoint_metrics.csv)，其中三种方法各自的top5也以JSON保存。

## 6. 输出

- [`projection_triples.jsonl`](processed/projection_triples.jsonl)
- [`lyapunov_by_token_window.csv`](processed/lyapunov_by_token_window.csv)
- [`lyapunov_by_token_overall.csv`](processed/lyapunov_by_token_overall.csv)
- [`lyapunov_checkpoint_summary.csv`](processed/lyapunov_checkpoint_summary.csv)
- [`lyapunov_last256_summary.csv`](processed/lyapunov_last256_summary.csv)
- [`period_checkpoint_summary.csv`](processed/period_checkpoint_summary.csv)
- [`period_by_token.csv`](processed/period_by_token.csv)
- [`period_recurrence_rows.csv`](processed/period_recurrence_rows.csv)
- [`checkpoint_loss.csv`](processed/checkpoint_loss.csv)
- [`window_endpoint_metrics.csv`](processed/window_endpoint_metrics.csv)
- [`lyapunov_by_checkpoint.png`](figures/lyapunov_by_checkpoint.png)
