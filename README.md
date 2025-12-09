  # 三维有限差分求解器

  ## 版本
  Ver 1.0 20251209 最基本程序

  ## 控制方程

  N-S方程守恒形式为：
  $$
  \frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} + \frac{\partial \mathbf{H}}{\partial z} = \frac{\partial \mathbf{F}_v}{\partial x} + \frac{\partial \mathbf{G}_v}{\partial y} + \frac{\partial \mathbf{H}_v}{\partial z} 
  $$

  守恒量和无粘通量为

  $$
  \mathbf U=\begin{bmatrix}
\rho\\ \rho u \\ \rho v \\ \rho w \\ E
\end{bmatrix},
\mathbf F=\begin{bmatrix}
\rho u\\ \rho u^2+p \\ \rho uv \\ \rho uw \\ u(E+p)
\end{bmatrix},
\mathbf G=\begin{bmatrix}
\rho v\\ \rho uv \\ \rho v^2+p \\ \rho vw \\ v(E+p)
\end{bmatrix},
\mathbf H=\begin{bmatrix}
\rho w\\ \rho uw \\ \rho vw \\ \rho w^2+p \\ w(E+p)
\end{bmatrix}
  $$

  其中总能$E=e+\frac{1}{2}(u^2+v^2+w^2)$。对于理想气体，内能为$e=\frac{p}{\rho(\gamma -1)}$
  
  粘性通量为

  $$
\mathbf{F}_{v} = \begin{bmatrix}
0 \\
\tau_{xx} \\
\tau_{xy} \\
\tau_{xz} \\
u\tau_{xx} + v\tau_{xy} + w\tau_{xz} - q_x
\end{bmatrix}
  $$

  $$
\mathbf{G}_{v} = \begin{bmatrix}
0 \\
\tau_{yx} \\
\tau_{yy} \\
\tau_{yz} \\
u\tau_{yx} + v\tau_{yy} + w\tau_{yz} - q_y
\end{bmatrix}
  $$

  $$
\mathbf{H}_{v} = \begin{bmatrix}
0 \\
\tau_{zx} \\
\tau_{zy} \\
\tau_{zz} \\
u\tau_{zx} + v\tau_{zy} + w\tau_{zz} - q_z
\end{bmatrix}
  $$

其中粘性应力为
$$
\begin{align*}
\tau_{xx} &= \mu \left( 2\frac{\partial u}{\partial x} - \frac{2}{3} \nabla \cdot \mathbf{V} \right) \\
\tau_{yy} &= \mu \left( 2\frac{\partial v}{\partial y} - \frac{2}{3} \nabla \cdot \mathbf{V} \right) \\
\tau_{zz} &= \mu \left( 2\frac{\partial w}{\partial z} - \frac{2}{3} \nabla \cdot \mathbf{V} \right) \\
\tau_{xy} = \tau_{yx} &= \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right) \\
\tau_{xz} = \tau_{zx} &= \mu \left( \frac{\partial u}{\partial z} + \frac{\partial w}{\partial x} \right) \\
\tau_{yz} = \tau_{zy} &= \mu \left( \frac{\partial v}{\partial z} + \frac{\partial w}{\partial y} \right)
\end{align*}
$$

热通量分量由傅里叶导热定律给出
$$
\begin{align*}
q_x &= -k \frac{\partial T}{\partial x} \\
q_y &= -k \frac{\partial T}{\partial y} \\
q_z &= -k \frac{\partial T}{\partial z}
\end{align*}
$$

对于高温或变粘度流动，粘性系数 $\mu$ 通常是温度的函数。常用的Sutherland公式为：

$$
\frac{\mu}{\mu_0} = \left( \frac{T}{T_0} \right)^{3/2} \frac{T_0 + S}{T + S}
$$
其中：
- $\mu_0$ 是参考温度 $T_0$ 下的粘度（对于空气，$\mu_0=1.716\times 10^{-5}[\mathrm{kg/ms}]$，$T_0=273.15[\mathrm K]$，一般认为对$T<2000K$有效）
- $S$ 是Sutherland常数（对于空气，$S = 110.4$ K）

  ### 无量纲化

  采用OpenCFD-SC类似方法无量纲化

  速度、密度、温度、长度、粘性系数分别使用参考速度$U_\text{ref}$，参考密度$\rho_\text{ref}$，参考温度$T_\text{ref}$，参考长度$L_\text{ref}$和参考粘性系数$\mu_\text{ref}$无量纲化。压力用动压$\rho_\text{ref}U_\text{ref}^2$无量纲化。

  无量纲化后方程含有无量纲参数Re（Reynolds数）, Ma（Mach数），计算方法为

  $$
    \text{Re}=\frac{\rho_\text{ref}U_\text{ref}L_\text{ref}}{\mu_\text{ref}} \\
    \text{Ma}=\frac{U_\text{ref}}{c_\text{ref}} = \frac{U_\text{ref}}{\sqrt{\gamma\tilde R T_\text{ref}}} \\
    \text{Pr}=\frac{c_p \mu_\text{ref}}{k}
  $$
  对于空气
  $$
  \tilde R = \frac{R_0}{\tilde M}= \frac{8.314 [\mathrm{J/(mol\cdot K)}]}{0.02896 [\mathrm{J/(mol)}]} = 287.08 [\mathrm K]
  $$

  化简后的无量纲方程为
  $$
    \rho \left[\frac{\partial \mathbf U}{\partial t} + \left(\mathbf U \cdot \nabla\mathbf U\right)\right] = -\nabla p + \mathrm{Re} \nabla \cdot \left[ \mu \left( (\nabla \mathbf U)+(\nabla \mathbf U)^T -\frac{2}{3}(\nabla \cdot \mathbf U)\mathbf I\right)\right]
  $$

  $$
    \rho \left[ \frac{\partial T}{\partial t} + (\mathbf{U} \cdot \nabla) T \right] = (\gamma - 1) \operatorname{Ma}^2 \left[ \frac{\partial p}{\partial t} + (\mathbf{U} \cdot \nabla) p \right] 
    + \frac{1}{\operatorname{Re} \Pr} \nabla \cdot (k \nabla T) \\
    + \frac{(\gamma - 1) \operatorname{Ma}^2 \mu}{2 \operatorname{Re}} 
    \left[ (\nabla \mathbf{U}) + (\nabla \mathbf{U})^{\mathrm{T}} - \frac{2}{3} (\nabla \cdot \mathbf{U}) \mathbf{I} \right] : 
    \left[ (\nabla \mathbf{U}) + (\nabla \mathbf{U})^{\mathrm{T}} - \frac{2}{3} (\nabla \cdot \mathbf{U}) \mathbf{I} \right]
  $$

  ## 数值方法