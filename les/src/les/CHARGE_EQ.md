# LES 电荷平衡模式（charge_eq）

## 功能概览

`charge_eq` 在 LES 的局域电荷预测之后，加入一次基于 Ewald Coulomb 矩阵的受约束电荷平衡求解。

局域网络及可选逐元素基线先给出参考电荷 $q_r$。程序再求解：

$$
\min_q \frac{1}{2}q^T A q + w(q-q_r)^T(q-q_r),
\qquad \mathbf{1}^Tq=S.
$$

$A$ 是当前 LES Ewald 电荷能量对电荷的 Hessian，定义为：

$$
E_{\mathrm{lr}}(q)=\frac{1}{2}q^TAq.
$$

$w$ 是 `regularization_weight`。较大的 $w$ 会让结果更接近网络参考电荷 $q_r$；较小的 $w$ 会让 Coulomb 能量对最终电荷的影响更强。代码通过 KKT 线性方程求解，解出的 $q$ 严格满足总电荷约束。

正则项用于确定求解出的电荷，不会额外加到 LES 返回的 `E_lr` 中。返回的 `latent_charges` 是求解后的 $q$，`E_lr` 是仅由 $q$ 和 Ewald 矩阵计算的静电能。矩阵、线性方程解和能量均保留自动微分计算图。

## 配置与用法

在 DeepMD 模型的 `les_params` 中设置：

```json
"les_params": {
  "local_charge": true,
  "charge_eq": true,
  "charge_eq_debug": true,
  "q_nn_debug": true,
  "e_lr_graph_debug": true,
  "regularization_weight": 0.1,
  "sigma": 1.0,
  "dl": 2.0
}
```

此处展示的是 `les_params` 内容；其余 DeepMD 模型配置照常填写。

- `charge_eq`：是否启用电荷平衡，默认 `false`。
- `charge_eq_debug`：启用时按 `log_freq` 在 DEBUG 级别分别输出求解前参考电荷 $q_r$ 和求解后电荷 $q$ 的逐元素均值与标准差；需要同时启用 `charge_eq`。
- `q_nn_debug`：在 DEBUG 级别输出 Q NN 参数范数摘要。
- `e_lr_graph_debug`：在 DEBUG 级别输出 `E_lr` 的 autograd graph 信息。
- `verbose`：INFO 级别输出最终电荷的逐元素均值与方差，以及数值形式的 `E_lr`。日志仅显示数值，不包含 Tensor 的 dtype、device 或 graph；不再输出跨元素汇总的电荷均值和标准差。
- `regularization_weight`：正则权重 $w$，默认 `0.1`；启用模式时必须大于零。
- `local_charge`：通过描述符预测 $q_r$。旧配置名 `use_atomwise` 仍可使用。
- `use_fixed_charges`：可作为局域网络输出上的逐元素基线，支持与 `charge_eq` 同用。
- `initial_guess`：可作为逐元素参考电荷基线，支持与 `charge_eq` 同用；长度必须等于 `type_map` 的类型数。

启用 `charge_eq` 但没有设置显式总电荷约束时，默认令 $S=0$。也可指定：

```json
"claim_total_charge": 0
```

或用 `claim_neutral: true` 表示中性。非零净电荷可通过 `claim_total_charge` 指定，例如 `claim_total_charge: 1`。`claim_neutral` 与 `claim_total_charge` 不能同时设置。每个 batch frame 独立构造矩阵并施加对应的总电荷约束。

## 代价与规模

矩阵 $A$ 的倒空间求和把 $k$ 当作收缩指标，用积化和差写成两次 GEMM：

$$
A=\big(\cos(\mathbf{r}k^T)\odot p\big)\cos(\mathbf{r}k^T)^T+\big(\sin(\mathbf{r}k^T)\odot p\big)\sin(\mathbf{r}k^T)^T .
$$

这样中间量只有 $O(nM)$，而不是按 $[n,n,M]$ 显式构造相位差时的 $O(n^2M)$。

$M$ 只由 cell 和 `dl` 决定，与原子数、坐标无关：水界面 slab（25.6 x 25.6 x 65 Å，`dl=2.0`）对应 $M=11146$。

对同一个 1566 原子的 slab，旧的显式写法需要两张 218 GB 的中间张量（float64 合计 407 GiB），任何单卡都放不下；收缩写法下同尺寸单次计算的峰值内存约 1.1 GiB。

收缩形式与原定义在数学上严格等价，数值上也是：与 $[n,n,M]$ 直接定义相比，float64 的逐元素相对误差为 1.6e-14，float32 为 3.7e-06；与 `compute_potential_triclinic` 的能量相比，$n=1566$ 时相对误差为 1.6e-15。

## 实现位置

- `les.py`：解析 `charge_eq` 与 `regularization_weight`；在每个 frame 上生成 Ewald 矩阵、求出最终电荷并计算 `E_lr`。未启用此选项时，原有 LES 前向路径不变。
- `module/ewald.py`：`Ewald.compute_coulomb_matrix(r, cell)` 返回满足 $E=\frac12q^TAq$ 的矩阵。周期情形使用 LES 当前的倒格点网格、截断、Gaussian 展宽和自相互作用约定，并按上一节的收缩形式装配；非周期情形使用 LES 的实空间电荷核。
- `module/charge_eq.py`：`project_zero_mean(q_r, A, w, total_charge=0.0)` 组装并求解 KKT 系统。函数名沿用原有名称；现在也支持非零 `total_charge`。
- `tests/test_charge_eq.py`：覆盖周期/非周期矩阵能量一致性、自相互作用开关、总电荷约束、多 frame 与梯度回传；另有两个回归用例固定倒空间收缩——在真实 $M$ 下与 $[n,n,M]$ 定义逐元素比对（float64 与 float32），并在 1566 原子的生产尺寸上验证矩阵有限、对称且能量正确。

## 兼容性与限制

- 与 `local_charge` / `use_atomwise` 兼容；$q_r$ 也可叠加 `use_fixed_charges` 或 `initial_guess` 基线。
- 与 `claim_total_charge` / `claim_neutral` 兼容。启用 `charge_eq` 时，总电荷约束在 KKT 求解中直接施加，不再执行原有的均匀电荷投影。
- 与 `freeze_charge` 不兼容，因为该模式固定电荷而 `charge_eq` 会重新优化电荷。
- 当前 `charge_eq` 只实现 charge-charge 的 Ewald 矩阵能量。若输入含 `latent_dipoles`、`latent_quads`、`latent_kappas` 或 `latent_alphas`，会报错，而不会静默忽略这些多极或响应项。
- `use_fixed_charges` 仍使用 LES 已有的元素序数映射；若 `type_map` 中有映射表不支持的自定义元素符号，该基线功能不可用。
- `charge_eq` 的求解依赖非奇异 KKT 系统。`regularization_weight` 必须为正；实际训练中仍需结合 Ewald 的能量尺度选择合适权重。
- 该路径已验证可进行 TorchScript 编译和前向/反向运行。完整 pytest 用例需要在安装了 `pytest` 的环境中执行。
