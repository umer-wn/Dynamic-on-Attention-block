# Pythia 早期训练 single-token Frobenius 扫描阶段报告

状态：`needs_more_loss_points`  
更新时间：由 `plot_pythia_early_single_token_scan.py` 自动生成

## 1. 研究问题

在固定 Pythia-70M 架构、固定 16 个 token 和固定 G1 single-token feedback 算子后，精确 token-level normalized Frobenius 是否随 attention 模型的训练程度和固定测试集 loss 发生可复现变化。

训练 checkpoint 没有在本实验中继续训练。`stepN` 是 Pythia 官方预训练过程中保存的静态权重；对每个静态 checkpoint 分别执行 test loss 和隐藏状态循环。

## 2. 当前完成度

- loss checkpoint 数：29
- 最低观测 loss：5.131336（step97000）
- 自适应搜索状态：`needs_more_loss_points`
- 已确认显著反转数：0
- 完整 dynamics checkpoint 数：29

本阶段实际运行缓存 fallback checkpoint `step0/step1000/step16000`。原定 `step5000` 从官方 Hub 预取因服务器到 `huggingface.co:443` connection timeout 失败；因此本阶段只作为方法、成本和混杂因素门控，不替代前 100 checkpoint 粗扫描。

## 3. 固定公式与数据流

每个 checkpoint 的 test loss 是固定 WikiText-2 test 前 128 个非空样本上的 token-weighted causal cross-entropy。single-token dynamics 为 `x_(t+1)=F_theta(x_t)`，输入输出均为 `[512]`；模型权重固定，LM head、softmax 和 token sampling 不进入循环。精确 Jacobian shape 为 `[512,512]`，主指标是 `||J||_F/sqrt(512)`。

为避免把训练权重变化与末端吸引子混在一起，指标拆成：当前 checkpoint 自身 embedding 上的 `self_t0`、迭代末端的 `tail_t767`，以及所有 checkpoint 在固定 step1000 token-vector bank 上求导的 `common_step1000_state`。

## 4. 当前数值摘要

| checkpoint | condition | median | q25 | q75 | std |
|---|---|---:|---:|---:|---:|
| step0 | common_step1000_state | 32.59745564 | 30.61762193 | 33.53476042 | 2.33855655 |
| step1000 | common_step1000_state | 33.09481543 | 30.71665084 | 35.45488520 | 3.89292702 |
| step5000 | common_step1000_state | 30.40947450 | 28.17298671 | 36.78696230 | 6.44135283 |
| step9000 | common_step1000_state | 31.76325824 | 27.78399153 | 34.74251309 | 6.54998486 |
| step10000 | common_step1000_state | 31.56863421 | 27.41063856 | 35.50202021 | 8.02556694 |
| step13000 | common_step1000_state | 31.56122176 | 29.32990893 | 36.18265405 | 8.15628462 |
| step16000 | common_step1000_state | 32.07207528 | 27.52350445 | 36.39285218 | 8.33336867 |
| step17000 | common_step1000_state | 36.52857650 | 30.56363620 | 40.95933711 | 9.73403825 |
| step21000 | common_step1000_state | 32.87384186 | 26.61148711 | 36.61839649 | 11.65618024 |
| step25000 | common_step1000_state | 30.49585733 | 27.10863515 | 40.89362040 | 11.15104946 |
| step29000 | common_step1000_state | 34.76601148 | 27.36741477 | 41.38097554 | 10.49854349 |
| step33000 | common_step1000_state | 32.27139818 | 29.24157521 | 39.31734850 | 6.41607667 |
| step37000 | common_step1000_state | 34.05482623 | 28.13177586 | 38.86192718 | 7.31438977 |
| step41000 | common_step1000_state | 41.94765644 | 33.56182342 | 47.66858064 | 29.82458020 |
| step45000 | common_step1000_state | 33.90168414 | 29.15963499 | 38.24974590 | 10.31701650 |
| step49000 | common_step1000_state | 32.74895502 | 28.04396678 | 39.00357768 | 23.97972422 |
| step53000 | common_step1000_state | 34.10976414 | 32.46451707 | 39.32533954 | 12.44614198 |
| step57000 | common_step1000_state | 38.85428276 | 33.17532333 | 46.10939810 | 8.31943968 |
| step61000 | common_step1000_state | 39.09342735 | 32.96756701 | 44.10731760 | 7.72846572 |
| step65000 | common_step1000_state | 34.62641977 | 26.96005773 | 38.37485662 | 40.39540479 |
| step69000 | common_step1000_state | 50.48236224 | 38.07189833 | 55.08844633 | 10.74366804 |
| step73000 | common_step1000_state | 49.24868285 | 41.85973811 | 57.21702537 | 10.62379441 |
| step77000 | common_step1000_state | 49.39485486 | 46.16853856 | 53.90665139 | 21.23177079 |
| step81000 | common_step1000_state | 47.38325659 | 44.69578404 | 59.76136385 | 25.97168179 |
| step85000 | common_step1000_state | 49.54881157 | 44.10386358 | 58.10493606 | 8.64666558 |
| step89000 | common_step1000_state | 51.23301312 | 43.41979191 | 61.24243783 | 19.92471920 |
| step93000 | common_step1000_state | 53.39081578 | 47.67436386 | 64.67102259 | 14.28375217 |
| step97000 | common_step1000_state | 48.53338946 | 42.63948313 | 54.71969984 | 11.46029892 |
| step100000 | common_step1000_state | 49.84802047 | 45.48312156 | 59.36259667 | 11.76017892 |
| step0 | self_t0 | 37.12347690 | 35.60914738 | 38.76632464 | 1.96966889 |
| step1000 | self_t0 | 33.09481543 | 30.71665084 | 35.45488520 | 3.89292702 |
| step5000 | self_t0 | 14.39508887 | 13.41484826 | 16.31168612 | 2.41954574 |
| step9000 | self_t0 | 12.93744180 | 11.27141835 | 15.04824735 | 2.34762741 |
| step10000 | self_t0 | 12.81180373 | 11.57805252 | 14.26042328 | 2.43580891 |
| step13000 | self_t0 | 12.28266455 | 11.86974412 | 13.76940070 | 2.02230590 |
| step16000 | self_t0 | 12.27088029 | 11.64855273 | 14.23520160 | 2.28114426 |
| step17000 | self_t0 | 13.38237765 | 12.67808965 | 14.97439461 | 1.72550879 |
| step21000 | self_t0 | 12.66768309 | 11.82561704 | 14.47180895 | 3.33619197 |
| step25000 | self_t0 | 12.11951983 | 11.69488696 | 14.79792912 | 2.11367164 |
| step29000 | self_t0 | 13.08442775 | 11.59859270 | 16.83412396 | 3.73693317 |
| step33000 | self_t0 | 14.13401915 | 12.78362401 | 16.75215711 | 3.21754768 |
| step37000 | self_t0 | 13.47118342 | 12.79670572 | 16.10047377 | 3.98203110 |
| step41000 | self_t0 | 16.24457553 | 15.04366953 | 18.36167984 | 3.98475593 |
| step45000 | self_t0 | 15.23153860 | 14.46828716 | 17.11419853 | 2.84157784 |
| step49000 | self_t0 | 14.92465829 | 13.46755340 | 16.61301788 | 2.57859046 |
| step53000 | self_t0 | 16.29002871 | 15.03872553 | 18.32394964 | 4.43221041 |
| step57000 | self_t0 | 18.41491299 | 17.44199447 | 19.55410556 | 3.56939750 |
| step61000 | self_t0 | 20.38364286 | 18.32928543 | 22.47192278 | 4.57655625 |
| step65000 | self_t0 | 16.43737342 | 14.92377961 | 18.59793619 | 2.39713234 |
| step69000 | self_t0 | 24.87593586 | 22.34573916 | 25.78603332 | 3.54775425 |
| step73000 | self_t0 | 25.29605834 | 23.60753555 | 29.24718243 | 4.67749846 |
| step77000 | self_t0 | 24.67411382 | 24.43919672 | 28.52948633 | 4.36347569 |
| step81000 | self_t0 | 27.58734984 | 25.18288899 | 30.90614442 | 4.36320411 |
| step85000 | self_t0 | 28.41007386 | 26.96221497 | 31.45999245 | 5.69904503 |
| step89000 | self_t0 | 28.14307189 | 27.16692255 | 34.06577969 | 6.37470081 |
| step93000 | self_t0 | 31.04940932 | 28.52795623 | 34.09908649 | 6.65859865 |
| step97000 | self_t0 | 30.57489784 | 27.78702003 | 34.43012874 | 8.07248174 |
| step100000 | self_t0 | 31.63455188 | 29.53629497 | 37.01437928 | 8.52298097 |
| step0 | tail_t767 | 0.68383236 | 0.67872430 | 0.68947982 | 0.00806698 |
| step1000 | tail_t767 | 0.68526091 | 0.68368086 | 0.69377364 | 0.00580772 |
| step5000 | tail_t767 | 0.68044020 | 0.67896976 | 0.68199428 | 0.00357238 |
| step9000 | tail_t767 | 0.69454767 | 0.69446709 | 0.69468938 | 0.00018303 |
| step10000 | tail_t767 | 0.68740817 | 0.67646029 | 0.69844047 | 0.01349666 |
| step13000 | tail_t767 | 0.67061688 | 0.67061684 | 0.67061688 | 0.00000003 |
| step16000 | tail_t767 | 0.68289458 | 0.68289455 | 0.68289464 | 0.00000023 |
| step17000 | tail_t767 | 0.71209486 | 0.71190259 | 0.71232698 | 0.00034327 |
| step21000 | tail_t767 | 0.70775325 | 0.69442073 | 0.72193980 | 0.02132617 |
| step25000 | tail_t767 | 0.67702706 | 0.67185684 | 0.68875301 | 0.01550385 |
| step29000 | tail_t767 | 0.68098880 | 0.67348885 | 0.69131132 | 0.01068123 |
| step33000 | tail_t767 | 0.69005973 | 0.67756580 | 0.71270340 | 0.02216560 |
| step37000 | tail_t767 | 0.70209078 | 0.68772091 | 0.71065186 | 0.01739580 |
| step41000 | tail_t767 | 0.66270727 | 0.65946903 | 0.68078198 | 0.01885831 |
| step45000 | tail_t767 | 0.68039156 | 0.65950999 | 0.69066359 | 0.02974907 |
| step49000 | tail_t767 | 0.65884489 | 0.65883313 | 0.69064493 | 0.02456991 |
| step53000 | tail_t767 | 0.62936045 | 0.62229120 | 0.63314269 | 0.01603340 |
| step57000 | tail_t767 | 0.65085650 | 0.64816240 | 0.65628966 | 0.01181912 |
| step61000 | tail_t767 | 0.59145928 | 0.59145928 | 0.59145928 | 0.00000001 |
| step65000 | tail_t767 | 0.57599114 | 0.57599114 | 0.57599114 | 0.00000003 |
| step69000 | tail_t767 | 0.49758801 | 0.49758801 | 0.49758801 | 0.00000001 |
| step73000 | tail_t767 | 0.40121956 | 0.40121954 | 0.40121958 | 0.00000003 |
| step77000 | tail_t767 | 0.37715908 | 0.37715908 | 0.37715908 | 0.00000000 |
| step81000 | tail_t767 | 0.38426618 | 0.38426618 | 0.38426618 | 0.00000000 |
| step85000 | tail_t767 | 0.39390643 | 0.39390642 | 0.39390643 | 0.00000002 |
| step89000 | tail_t767 | 0.41369673 | 0.41369673 | 0.41369673 | 0.00000000 |
| step93000 | tail_t767 | 0.42714874 | 0.42714874 | 0.42714874 | 0.00000000 |
| step97000 | tail_t767 | 0.43606255 | 0.43606255 | 0.43606255 | 0.00000000 |
| step100000 | tail_t767 | 0.44009507 | 0.44009503 | 0.44009507 | 0.00000002 |

## 5. 图表阅读

- [checkpoint_test_loss.png](/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/figures/checkpoint_test_loss.png)：蓝色粗扫描、橙色自适应真实 checkpoint、灰色边界 sentinel。连线只帮助阅读，不代表数值插值。
- [adjacent_loss_bootstrap.png](/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/figures/adjacent_loss_bootstrap.png)：相邻实测 checkpoint 的 loss 差及 paired-bootstrap 95% CI；跨过零线表示统计未决，跨过虚线还需满足 1% PPL 的实际效应阈值。
- [checkpoint_normalized_frobenius.png](/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/figures/checkpoint_normalized_frobenius.png)：灰色为 16 个 token，蓝线为中位数、带为 IQR；虚线 1 是 identity RMS-gain 参考，不是 Lyapunov 零线。
- [loss_vs_normalized_frobenius.png](/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/figures/loss_vs_normalized_frobenius.png)：检验 Frobenius 更接近模型能力还是只随 step 漂移。需要联合 token 配对与反转带阅读，不能只凭 checkpoint 均值拟合。
- [frobenius_state_weight_controls.png](/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/figures/frobenius_state_weight_controls.png)：并列比较 `self_t0/tail_t767/common-state`，用来识别末端吸引子选择的混杂。
- 轨迹 grid：每个 panel 是一个 checkpoint 下单个 token 的 `(z0,z1)` 轨迹；黑叉是初始 embedding，红点是末端。固定轴和固定投影保证可比，但二维重叠不证明高维状态相同。

## 6. 当前结论边界

当前 loss 在两个区间均显著下降，但仅有三个非均匀 checkpoint，不能计算可信的训练相关性或寻找反升段。`tail_t767` 在 step16000 的 token 标准差约 `2.28e-7`，说明不同 token 很可能进入同一末端状态/吸引子；因此仅看 tail Frobenius 会把训练与吸引子选择混合。

在自适应搜索和完整粗扫描完成前，不下“存在真实关系”的结论。最终支持条件包括：大多数 token 配对方向一致、checkpoint 相关 CI 不跨零、leave-one-token-out 稳健，并在 loss 下降后反升的区间出现与 loss 相符的 Frobenius 回转。即使通过，也只是 checkpoint 间关联，不是训练因果；G1 的 seq1 attention 已退化，不能直接推广到完整上下文 attention Jacobian。

## Canonical organized index

本实验的整理后索引路径：

```text
/data1/luohaoming/model_feature/experiments_ordered/09_pythia_early_training_frobenius_scan/
```

原始大数据仍保留在：

```text
/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/
```

整理策略为复制小文件与 manifest 引用大数据，不删除旧路径。

