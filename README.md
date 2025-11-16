**简介**
- **项目**: `ns3d_cpp` — 一个基于有限体积/有限差分混合实现的可压缩三维 Navier–Stokes 求解器原型（C++17 + MPI）。
- **目的**: 研究和验证重构格式、粘性/不可粘通量分离、并行域分解与输出（Tecplot）。

**主要功能**
- **数值格式**: 有限体积（面通量）配合高阶差分用于粘性项。
- **重构器**: 支持多种面重构（WENO5、MDCD、线性等）和可选的特征/分量方向重构。
- **时域积分**: 3-stage Runge–Kutta（显式）时间推进。
- **并行**: MPI Cartesian 分解，支持进程间周期交换（按面打包/Sendrecv）与本地周期复制。
- **I/O**: 每个 rank 输出 Tecplot ASCII 文件；仓库含 `tools/merge_tecplot.py` 用于合并分片文件。
- **诊断**: 记录能量与残差指标，输出步写入文件。

**快速开始（构建）**
- 先准备 `build` 目录并使用 CMake：

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j4
```

- 成功后可执行文件为 `./build/ns3d`。

**运行（单进程 smoke test）**
- 单进程启动并检查初始化：

```bash
./build/ns3d
```
- 程序会打印初始化信息并在当前目录写出类似 `initial_field_rank0.dat` 的 Tecplot 文件。

**运行（MPI 多进程）**
- 使用 `mpirun` / `mpiexec` 启动（示例 2 进程）：

```bash
mpirun -n 2 ./build/ns3d
```

- 每个 rank 会写出 `initial_field_rank<rank>.dat`，后续按 `output_freq` 写出时步数据。

**MPI 注意事项与常见问题**
- 程序会建立 Cartesian communicator（MPI_Cart_create）。若出现 `MPI_Dims_create` / `MPI_ERR_DIMS` 错误，通常是传入的 `dims` 未初始化或不一致。解决办法：
  - 确保每个进程都一致地调用初始化函数；当前代码在创建前会将 `dims` 置为 `{0,0,0}`，由 MPI 自动计算合适的分解。
- 周期边界：当进程邻居存在时，代码使用按面打包的 `MPI_Sendrecv`（包含 `rho, rhou, rhov, rhow, E, u, v, w, p, T`）交换内点层；若邻居为 `MPI_PROC_NULL` 则回退为本地周期复制（`apply_periodic_bc`）。

**主要输入/控制**
- `solver.in`：文本格式的参数文件（物理参数、重构器选择、边界类型、网格大小、ghost 层数等）。在仓库中有示例默认值。

**文件与目录概览**
- `main.cpp`：程序入口，初始化、边界、IO 与时间推进调用。
- `field_structures.h`：网格/域分解、`Field3D` 数据结构、打包/通信辅助函数。
- `initialization.cpp`：读取 `solver.in` 并初始化场变量。
- `boundary.cpp`：边界条件实现（wall/symmetry/inflow/outflow/periodic）；包含 MPI 周期交换逻辑。
- `fluxCalculation.cpp` / `InviscousFlux.cpp` / `ViscousFlux.cpp`：不粘/粘性通量计算、梯度与重构相关核。
- `reconstruction.cpp`：重构算子实现（WENO、MDCD、线性等）。
- `range_kutta.cpp`：RHS 组装与 Runge–Kutta 时步。
- `output.cpp`：Tecplot 写出函数。
- `tools/merge_tecplot.py`：合并多 rank 的 Tecplot 文件为单一文件（Python 3）。

**运行输出**
- 输出文件名约定：`initial_field_rank<N>.dat`、`output_step_XXXXX_rank<N>.dat`。
- 用 `tools/merge_tecplot.py` 合并：

```bash
python3 tools/merge_tecplot.py output_step_00200_rank*.dat -o merged_step_00200.dat
```

**开发者说明 / 后续工作**
- 当前 TODO 项目（示例）:
  - 移除或重构持久 `rhs_*` 中间数组（用户希望 RHS 仅为临时中间量）。
  - 继续完善特征方向重构与 C6th 梯度在并行边界的严格验证。
- 性能提示：打包/通信可改为持久请求（`MPI_Send_init`/`MPI_Recv_init`）并重用缓冲区；可将通信与内部计算重叠以提高并行效率。

**联系方式 / 贡献**
- 代码仓库中有注释，欢迎修改并提交 PR。遇到运行问题请把 `cmake` 输出、`mpirun` 错误和 `solver.in` 内容一并发送以便快速定位。

---
(此 README 简要概述项目结构与使用流程；如需更详尽的设计文档或数值公式说明，我可以基于当前代码补充更深入的开发文档。)

**设计与数值方法（详细）**

1) 守恒方程（可压缩 Navier–Stokes，保守型）

  - 状态变量向量：$$U = [\rho,\;\rho u,\;\rho v,\;\rho w,\;E]^T$$。
  - 控制方程（守恒型）：

  $$
  \frac{\partial U}{\partial t} + \nabla\cdot F(U) = \nabla\cdot F^{(v)}(U,\nabla U),
  $$

  其中 $F(U) = (F, G, H)$ 为无粘（对流）通量，$F^{(v)}$ 为粘性通量项。

  - 压强与声速：

  $$
  p = (\gamma - 1)\left(E - \tfrac12\rho (u^2+v^2+w^2)\right),\quad a=\sqrt{\gamma p/\rho}.
  $$

2) 有粘/无粘通量离散策略

  - 对流项使用有限体积思想：在单元 $i,j,k$ 上，右端项近似为面通量差分

  $$
  \mathrm{RHS}_{i,j,k} = -\frac{1}{\Delta V} \left( F_{i+\frac12}-F_{i-\frac12} + G_{j+\frac12}-G_{j-\frac12} + H_{k+\frac12}-H_{k-\frac12} \right).
  $$

  - 面通量 $F_{i+1/2}$ 由左右重构态 $U_{L},U_{R}$ 计算数值通量（FVS / Riemann/flux-splitting）。

  - 粘性项通过在体结点上计算应力/热通量（梯度算子），再对这些量做散度，代码中实现了 2/4/6 阶中心差分模板用于梯度计算。

3) Flux Vector Splitting (FVS) 与特征分解

  - 支持多种 FVS：Steger–Warming, Lax–Friedrichs, Van Leer（在 `SolverParams::FVS_Type` 中选择）。
  - 在特征空间进行重构（可选 `char_recon = true`），或直接在分量空间重构。

4) 重构算子（面重构）——以 WENO5 为例

  - WENO5 使用三个 3 点候选子模板 $S_0,S_1,S_2$，线性权重 $d_0,d_1,d_2$（例如 $d=[0.1,0.6,0.3]$，依具体实现而定），平滑度指标 $\\beta_i$：

  $$
  \beta_0 = \frac{13}{12}(f_{i-2}-2f_{i-1}+f_i)^2 + \frac{1}{4}(f_{i-2}-4f_{i-1}+3f_i)^2,
  $$

  类似地定义 $\beta_1,\beta_2$。非线性权重：

  $$
  \alpha_i = \frac{d_i}{(\varepsilon+\beta_i)^p},\quad \omega_i = \frac{\alpha_i}{\sum_j \alpha_j}.
  $$

  - 最终重构：$f_{i+1/2}^{-} = \sum_i \omega_i q_i(i+1/2)$，其中 $q_i$ 为第 $i$ 个子模板在面处的重构值。

  - 代码中还实现了线性、MDCD 以及 C6th 等重构/插值供对比。

5) 高阶导数/梯度模板（代码实现）

  - 二阶中心：$\dfrac{f_{i+1}-f_{i-1}}{2\,\Delta x}$。
  - 四阶中心：$\dfrac{-f_{i+2}+8f_{i+1}-8f_{i-1}+f_{i-2}}{12\,\Delta x}$。
  - 六阶中心（代码中采用）：

  $$
  \frac{-f_{i+3}+9f_{i+2}-45f_{i+1}+45f_{i-1}-9f_{i-2}+f_{i-3}}{60\,\Delta x}.
  $$

  - 近边界处采用向前/向后适配的低阶格式以保持稳定性（代码中根据到边界的距离选择差分阶次）。

6) 粘性应力與熱流（连续形式）

  - 黏性应力（牛顿流体）：

  $$
  \tau_{xx} = 2\mu\frac{\partial u}{\partial x} - \tfrac{2}{3}\mu(\nabla\cdot\mathbf{u}),\quad
  \tau_{xy} = \mu\left(\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}\right),\; \text{等等}.
  $$

  - 热通量（傅里叶律）： $\mathbf{q} = -k\nabla T$，其中 $k = \mu C_p / Pr$（代码用 $q_x = -\mu C_p/Pr\;\partial_x T$）。

7) 时间推进（3-stage Runge–Kutta）

  - 采用显式分段 RK（Shu–Osher 类型）示例公式：

  $$
  U^{(1)} = U^n + \Delta t\,\mathcal{R}(U^n),\\
  U^{(2)} = \tfrac34 U^n + \tfrac14\left(U^{(1)} + \Delta t\,\mathcal{R}(U^{(1)})\right),\\
  U^{n+1} = \tfrac13 U^n + \tfrac23\left(U^{(2)} + \Delta t\,\mathcal{R}(U^{(2)})\right),
  $$

  其中 $\mathcal{R}$ 为空间离散算子（面通量差分 + 粘性散度）。

8) 时间步长（CFL）

  - 局部稳定条件（显式）近似为：

  $$
  \Delta t = \text{CFL} \times \min\left(\frac{\Delta x}{|u|+a},\frac{\Delta y}{|v|+a},\frac{\Delta z}{|w|+a}\right)
  $$

9) 并行与 Halo/周期交换实现细节

  - 使用 MPI Cartesian communicator（`MPI_Cart_create`）进行域分解，局部信息在 `LocalDesc` 中保存（`nx,ny,nz,ngx,ngz,ngz`）。
  - halo/ghost 的交换：当邻居存在且边界为周期性时，采用按面打包 + `MPI_Sendrecv`/`MPI_Isend`-`MPI_Irecv` 方式交换内点层数据。打包顺序在实现中为沿 k->j->i 的确定性顺序，消息内按单元依次存放 $[\\rho,\\;\\rho u,\\;\\rho v,\\;\\rho w,\\;E,\\;u,\\;v,\\;w,\\;p,\\;T]$（以便重建 primitives 与 conserved）。
  - 若邻居为 `MPI_PROC_NULL`（局部边界），则对不同边界类型回退到本地处理：`apply_periodic_bc`（本地周期映射）、`apply_wall_bc`、`apply_outflow_bc` 等。

10) 数值稳定性与调试建议

  - 若遇到 NaN/发散：优先检查初始总能量 $E$ 的计算（确保内部能量为正）、Roe/平均态或通量分裂中使用的能量项（代码中曾修复 $H=(E+p)/\\rho$ 的相关错误）。
  - 逐步放宽分辨率/降阶（例如从 WENO5 切换为线性重构或从 C6th 降到 C4th）有助定位不稳定源。

--

如果你希望我把这些公式转为一份单独的设计文档（例如 `docs/DESIGN.md`），或者希望在源码注释中把关键公式与参考文献链接化，我可以把 README 中的这一节拆分出来并生成独立文档（并附参考文献条目）。
