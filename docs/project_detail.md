# 多无人机空中基站通信保障系统 —— 技术详解文档

> 状态说明（2026-04-04 校准）：本文档已根据当前仓库实现重新核对。特别是 MAPPO 的 critic 语义、训练日志格式、离线绘图链路与 attention/non-attention 分支说明，均已对齐当前代码，而非历史版本。

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 环境设计](#2-环境设计)
  - [2.1 场景模型](#21-场景模型)
  - [2.2 移动模型](#22-移动模型)
  - [2.3 通信模型](#23-通信模型)
  - [2.4 时延模型](#24-时延模型)
  - [2.5 能耗模型](#25-能耗模型)
  - [2.6 缓存模型](#26-缓存模型)
  - [2.7 奖励设计](#27-奖励设计)
- [3. 算法设计](#3-算法设计)
  - [3.1 观测空间](#31-观测空间)
  - [3.2 动作空间](#32-动作空间)
  - [3.3 支持的MARL算法](#33-支持的marl算法)
- [4. 仿真流程](#4-仿真流程)

---

## 1. 项目概述

本项目实现了一个基于**多智能体深度强化学习（MARL）**的多无人机空中基站通信保障系统。系统中多个无人机（UAV）作为空中基站，为地面和空中的用户设备（UE）提供通信服务，同时通过宏基站（MBS）进行内容回源。

### 核心优化目标

系统联合优化三个相互依赖的组件：

| 组件                      | 控制方式        | 说明                   |
| ------------------------- | --------------- | ---------------------- |
| **UAV 轨迹规划**    | MARL 智能体控制 | 3D 位置移动决策        |
| **3D 波束赋形控制** | MARL 智能体控制 | 定向天线波束指向       |
| **内容缓存策略**    | GDSF 自适应算法 | 基于请求频率的缓存更新 |

### 优化目标函数

$$
\text{maximize} \quad \alpha_3 \cdot \text{Fairness} + \alpha_{\text{rate}} \cdot \text{Throughput} - \alpha_1 \cdot \text{Latency} - \alpha_2 \cdot \text{Energy}
$$

---

## 2. 环境设计

### 2.1 场景模型

#### 基本参数

| 参数       | 符号                         | 默认值         | 说明         |
| ---------- | ---------------------------- | -------------- | ------------ |
| 区域大小   | $X_{\max} \times Y_{\max}$ | 1000m × 1000m | 仿真覆盖区域 |
| UAV 数量   | $U$                        | 10             | 空中基站数量 |
| UE 数量    | $M$                        | 100            | 用户设备数量 |
| 时隙长度   | $\tau$                     | 1.0s           | 单个决策周期 |
| 每回合步数 | $T$                        | 1000           | 训练回合长度 |

#### UAV 参数

| 参数         | 符号                      | 默认值 | 说明               |
| ------------ | ------------------------- | ------ | ------------------ |
| 最低飞行高度 | $H_{\min}$              | 100m   | UAV 高度下限       |
| 最高飞行高度 | $H_{\max}$              | 500m   | UAV 高度上限       |
| 飞行速度     | $v^{\text{UAV}}$        | 30 m/s | 3D 最大移动速度    |
| 覆盖半径     | $R$                     | 250m   | 3D 球形覆盖范围    |
| 感知范围     | $R^{\text{sense}}$      | 500m   | 邻居 UAV 发现距离  |
| 参考间距     | $d_{\min}^{\text{UAV}}$ | 100m   | 覆盖重叠约束参考值（当前安全判定由 50m 危险距离与 5m 碰撞距离驱动） |

#### UE 分层模型

系统支持地面和空中 UE 的混合场景：

| UE 类型 | 占比 | 高度范围   | 移动特性    |
| ------- | ---- | ---------- | ----------- |
| 地面 UE | 50%  | $z = 0$  | 2D 随机游走 |
| 空中 UE | 50%  | 50m - 600m | 3D 随机游走 |

#### MBS（宏基站）

- 位置：$(500, 500, 300)$ m（区域中心上空）
- 作为所有内容的回源节点
- 提供回程链路（Backhaul）连接

#### 系统约束与安全参数

| 参数类别 | 参数名 | 值 | 用途 |
|--------|-------|-----|------|
| **碰撞避免** | 危险距离下限（$d_{\text{danger}}$） | 50m | 开始线性惩罚的距离上界 |
| | 碰撞距离（$d_{\text{collision}}$） | 5m | 引发失效的距离下界 |
| | 危险靠近惩罚系数 | 4.0 | 最大惩罚 (当 $d_{\min}=5m$ 时) |
| | 碰撞失效惩罚 | 10.0 | 固定惩罚 (每次碰撞) |
| **边界约束** | 边界违规惩罚 | 2.0 | 越界动作的固定惩罚 |
| **内容缓存** | 更新周期 ($T_{\text{cache}}$） | 10 步 | GDSF 缓存策略更新间隔 |
| | 缓存文件数 | 20 | 可缓存内容数上限 |

---

### 2.2 移动模型

#### 2.2.1 UE 移动模型 —— 3D 随机游走（Random Waypoint）

UE 采用经典的**随机游走模型（Random Waypoint Model）**：

**算法流程：**

1. 随机生成目标位置 $(x_{\text{wp}}, y_{\text{wp}}, z_{\text{wp}})$
2. 随机生成到达后的等待时间 $t_{\text{wait}} \in [0, T_{\text{wait}}^{\max}]$
3. 以最大速度 $v^{\text{UE}}_{\max}$ 沿直线移动向目标
4. 到达后等待 $t_{\text{wait}}$ 个时隙
5. 重复步骤 1-4

**数学表达：**

$$
\vec{p}_{\text{UE}}(t+1) = \vec{p}_{\text{UE}}(t) + \min(d_{\max}^{\text{UE}}, \|\vec{p}_{\text{wp}} - \vec{p}_{\text{UE}}(t)\|) \cdot \frac{\vec{p}_{\text{wp}} - \vec{p}_{\text{UE}}(t)}{\|\vec{p}_{\text{wp}} - \vec{p}_{\text{UE}}(t)\|}
$$

**参数设置：**

- 最大移动距离：$d_{\max}^{\text{UE}} = 20$ m/时隙
- 最大等待时间：$T_{\text{wait}}^{\max} = 10$ 时隙

#### 2.2.2 UAV 移动模型 —— 智能体控制

UAV 的移动与波束控制完全由 MARL 智能体配置：

**动作解释：**

智能体根据不同配置输出连续动作向量：
若 `BEAM_CONTROL_ENABLED = True`，输出为 **5 维**向量 $\vec{a} = [a_x, a_y, a_z, a_\theta, a_\phi] \in [-1, 1]^5$；
若未启用，输出为 **3 维**移动向量 $\vec{a}_{\text{move}} = [a_x, a_y, a_z] \in [-1, 1]^3$。

系统对位移相关向量**模长**进行裁剪（保持方向不变）：

说明：这里采用的是**本地环境的有界动作契约**。动作首先被定义为归一化控制量，再映射到物理移动/波束角语义；这与官方 `onpolicy` 仓库中 Box 动作“同一动作张量直接用于执行与 log-prob 评估”的默认契约并不完全相同。

$$
\vec{\Delta p} = \frac{\vec{a}_{\text{move}}}{\max(\|\vec{a}_{\text{move}}\|, 1)} \times v^{\text{UAV}} \times \tau
$$

即动作向量的方向决定移动方向，模长（裁剪到最大为 1）决定移动距离占最大距离的比例。

**位置更新：**

$$
\vec{p}_{\text{UAV}}(t+1) = \text{clip}(\vec{p}_{\text{UAV}}(t) + \vec{\Delta p}, \vec{p}_{\min}, \vec{p}_{\max})
$$

**安全约束机制（当前实现）：**

系统不再使用“迭代推开式”碰撞消解，而是基于同时间同步轨迹计算最小间距并施加惩罚/失效：

1. 根据动作先得到每架 UAV 在单时隙内的 move-then-hover 轨迹；
2. 对每一对活跃 UAV 计算同步最小距离

$$
d_{\min}^{(i,j)} = \min_{t \in [0,\tau]} \left\|\vec p_i(t)-\vec p_j(t)\right\|_2
$$

3. 若 $d_{\min}^{(i,j)} \le 5\text{m}$，两者标记为碰撞失效（本回合后续时隙不再活跃）；
4. 若 $5\text{m} < d_{\min}^{(i,j)} < 50\text{m}$，施加连续危险靠近惩罚（线性插值）；
5. 越界动作会触发边界违规标记，位置按边界裁剪。

危险靠近惩罚项：

$$
r_{\text{unsafe}} = P_{\text{unsafe}} \cdot \text{clip}\!\left(\frac{50 - d_{\min}}{50 - 5}, 0, 1\right),\quad P_{\text{unsafe}}=4.0
$$

---

### 2.3 通信模型

通信模型是本系统的核心，实现了完整的 **3D 球形覆盖信道模型**，支持多种链路类型和先进的信道建模技术。

#### 2.3.1 链路类型概览

系统包含三类通信链路，使用不同的频段以避免相互干扰：

```
                         ┌─────────────────┐
                         │       MBS       │
                         │  (500,500,300)  │
                         └────────┬────────┘
                                  │
                    ══════════════╪══════════════  Backhaul Link
                    ║  B_backhaul = 40 MHz       ║  (UAV-MBS 回程链路)
                    ║  全双工、点对点            ║
                    ══════════════╪══════════════
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
    ┌───────┐                                           ┌───────┐
    │ UAV_1 │◄══════════════════════════════════════════│ UAV_2 │
    └───┬───┘        Inter-UAV Link (UAV-UAV)           └───┬───┘
        │            B_inter = 30 MHz                       │
        │            FDM 频分复用                            │
        │                                                   │
   ═════╪═════                                         ═════╪═════
   ║ Edge Link ║                                       ║ Edge Link ║
    ║ B_edge=30MHz                                      ║ B_edge=30MHz
   ║ OFDMA多址 ║                                       ║ OFDMA多址 ║
   ═════╪═════                                         ═════╪═════
        │                                                   │
        ▼                                                   ▼
    ┌───────┐  ┌───────┐                            ┌───────┐  ┌───────┐
    │  UE   │  │  UE   │   ... 同频干扰 ...         │  UE   │  │  UE   │
    └───────┘  └───────┘                            └───────┘  └───────┘
```

**频谱资源分配：**

| 链路类型                | 频段带宽 | 多址方式 | 特点                            |
| ----------------------- | -------- | -------- | ------------------------------- |
| Edge Link (UE-UAV)      | 30 MHz   | OFDMA    | 同一 UAV 的多个 UE 共享带宽     |
| Inter-UAV Link          | 30 MHz   | FDM      | 被多个 UAV 选为协作者时分割带宽 |
| Backhaul Link (UAV-MBS) | 40 MHz   | 点对点   | 全双工，区分上下行功率          |

---

#### 2.3.2 信道增益模型

信道增益模型综合考虑了路径损耗、视距概率和波束增益。

##### 2.3.2.1 仰角计算

仰角（Elevation Angle）是计算 LoS 概率的关键参数：

$$
\theta_{\text{elev}} = \arctan\left(\frac{|z_1 - z_2|}{\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}}\right)
$$

- 仰角范围：$[0°, 90°]$
- 垂直链路（正上方/正下方）：$\theta_{\text{elev}} = 90°$
- 水平链路：$\theta_{\text{elev}} \approx 0°$

##### 2.3.2.2 路径损耗模型

采用简化的**自由空间路径损耗模型**（Free Space Path Loss）：

$$
L(d) = d^2
$$

其中 $d$ 为发射端与接收端之间的 3D 欧氏距离：

$$
d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2}
$$

> **注**：此简化模型省略了载波频率相关项，将其合并到常数增益因子 $G_0 \cdot g_0$ 中。

##### 2.3.2.3 视距概率模型（LoS Probability）

视距概率决定了链路是 LoS（视距）还是 NLoS（非视距）。系统根据链路类型采用不同的模型：

**（A）地对空链路（Ground-to-Air）—— ITU-R / 3GPP 模型**

当链路中有一端高度低于 50m 时，使用 ITU-R P.1410 / 3GPP TR 36.777 模型：

$$
P_{\text{LoS}}(\theta) = \frac{1}{1 + a \cdot \exp\left(-b \cdot (\theta - a)\right)}, \quad \theta \text{ 以"度"为单位}
$$

其中 $\theta$ 是仰角（度），$(a, b)$ 是环境相关参数。

> **注**：本文采用 3GPP TR 36.777 推荐的 G2A LoS 概率模型。公式中参数 $a$ 同时出现在 Sigmoid 曲线的幅度因子和角度偏移项中，这是标准模型的原始形式。

**环境参数表：**

| 环境类型                  | 参数$a$ | 参数$b$ | 典型 LoS 概率曲线   | 适用场景             |
| ------------------------- | --------- | --------- | ------------------- | -------------------- |
| 郊区 (suburban)           | 4.88      | 0.43      | 仰角 30° 时 ≈100% | 开阔地带、低密度建筑 |
| 城市 (urban)              | 9.61      | 0.16      | 仰角 30° 时 ≈73%  | 普通城市街区         |
| 密集城区 (dense_urban)    | 12.08     | 0.11      | 仰角 30° 时 ≈37%  | 商业中心、高密度区   |
| 高层城区 (highrise_urban) | 27.23     | 0.08      | 仰角 30° 时 ≈4%   | 摩天大楼区域         |

**（B）空对空链路（Air-to-Air）**

当链路双方均在空中（高度 ≥ 50m）时，建筑物遮挡概率较低，采用简化模型：

$$
P_{\text{LoS}}^{\text{A2A}} = 0.8 + 0.2 \cdot \frac{\theta_{\text{elev}}}{90}, \quad \theta_{\text{elev}} \in [0°, 90°]
$$

其中 $\theta_{\text{elev}}$ 为仰角（单位：度）。该模型确保空对空链路的 LoS 概率在 $[0.8, 1.0]$ 范围内，仰角越大概率越高。相比更乐观的 0.9 下界，0.8 的设置更贴近常见空对空信道模型，考虑了远距离水平链路可能被高层建筑或地形遮挡的情况。

##### 2.3.2.4 综合信道增益公式

将路径损耗、LoS 概率和波束增益综合，得到信道增益：

$$
G_{\text{channel}} = \frac{G_0 \cdot g_0 \cdot G_{\text{beam}}}{\bar{L}(d)}
$$

其中**平均路径损耗** $\bar{L}(d)$ 考虑了 LoS 和 NLoS 两种情况：

$$
\bar{L}(d) = P_{\text{LoS}} \cdot L(d) + (1 - P_{\text{LoS}}) \cdot L(d) \cdot L_{\text{NLoS}}
$$

**参数说明：**

| 参数          | 符号                | 值                       | 说明                     |
| ------------- | ------------------- | ------------------------ | ------------------------ |
| 常数增益因子  | $G_0 \cdot g_0$   | $3.244 \times 10^{-4}$ | 包含载波频率、天线效率等 |
| NLoS 额外损耗 | $L_{\text{NLoS}}$ | 10 dB (线性值 10)        | 阴影衰落和绕射损耗       |
| 波束增益      | $G_{\text{beam}}$ | 见 2.3.3 节              | 仅用于 UAV-UE 链路       |

---

#### 2.3.3 3D 波束赋形模型（3GPP TR 38.901）

系统实现了完整的 **3D 波束赋形**（3D Beamforming）模型，UAV 配备定向天线，可以将信号能量集中在特定方向。

##### 2.3.3.1 球坐标系定义

波束方向使用**物理学球坐标系**表示：

```
                    Z+ (天顶, θ=0°)
                    │
                    │   ╱ 波束方向
                    │  ╱  
                    │ ╱ θ (俯仰角)
                    │╱___________
                    ┼────────────► Y+
                   ╱│
                  ╱ │
                 ╱  │
               X+   │
                    │
                    ▼ Z- (天底, θ=180°)
```

| 坐标   | 符号       | 范围                | 定义                        |
| ------ | ---------- | ------------------- | --------------------------- |
| 俯仰角 | $\theta$ | $[0°, 180°]$    | 从 Z+ 轴（天顶）测量的角度  |
| 方位角 | $\phi$   | $[-180°, 180°]$ | XY 平面内从 X+ 轴逆时针测量 |

**俯仰角的物理含义：**

- $\theta = 0°$：正上方（天顶方向）
- $\theta = 90°$：水平方向
- $\theta = 180°$：正下方（天底方向）

##### 2.3.3.2 波束方向单位向量

给定波束方向 $(\theta_0, \phi_0)$，其单位向量（在三个坐标轴的投影）为：

$$
\vec{v}_{\text{beam}} = \begin{pmatrix} \sin\theta_0 \cos\phi_0 \\ \sin\theta_0 \sin\phi_0 \\ \cos\theta_0 \end{pmatrix} = \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}
$$

##### 2.3.3.3 默认波束指向计算

当智能体不控制波束方向时，波束默认指向关联 UE 的**几何质心**：

$$
\vec{p}_{\text{centroid}} = \frac{1}{N_{\text{UE}}} \sum_{i=1}^{N_{\text{UE}}} \vec{p}_{\text{UE}}^{(i)}
$$

$$
\theta_0 = \arccos\left(\frac{dz}{\|\vec{d}\|}\right), \quad \phi_0 = \arctan2(dy, dx)
$$

其中 $\vec{d} = \vec{p}_{\text{centroid}} - \vec{p}_{\text{UAV}} = (dx, dy, dz)$。

##### 2.3.3.4 波束增益计算（3GPP 天线模型）

波束增益根据目标方向与波束指向之间的**角度偏差**（Angular Deviation）计算。

**Step 1：计算目标方向单位向量**

- $p_X$ 为 $X$ 的位置向量，$\vec{v}_{\text{target}}$ 为从无人机指向目标点的单位向量

$$
\vec{v}_{\text{target}} = \frac{\vec{p}_{\text{target}} - \vec{p}_{\text{UAV}}}{\|\vec{p}_{\text{target}} - \vec{p}_{\text{UAV}}\|}
$$

**Step 2：计算角度偏差（大圆距离）**

$\vec{v}_{\text{beam}}$ 为实际波束指向的单位向量，使用向量点积计算真实的 3D 角度偏差，避免极点附近的数值问题：

- 其中 $\vec{v}_{\text{target}}\cdot \vec{v}_{\text{beam}}=\cos\Delta\psi$

$$
\Delta\psi = \arccos(\vec{v}_{\text{beam}} \cdot \vec{v}_{\text{target}})
$$

**Step 3：计算增益衰减(dB)**

采用 3GPP TR 38.901 的天线模型：

$$
A(\Delta\psi) = \min\left(12 \cdot \left(\frac{\Delta\psi}{\psi_{\text{3dB}}}\right)^2, A_{\max}\right) \quad \text{[dB]}
$$

**Step 4：计算线性波束增益**

$$
G_{\text{beam}} = 10^{(G_{\max} - A(\Delta\psi)) / 10}
$$

**波束参数：**

| 参数         | 符号                  | 值     | 说明               |
| ------------ | --------------------- | ------ | ------------------ |
| 最大天线增益 | $G_{\max}$          | 18 dBi | 波束主瓣方向增益   |
| 3dB 波束宽度 | $\psi_{\text{3dB}}$ | 30°   | 功率下降一半的角度 |
| 旁瓣衰减上限 | $A_{\max}$ (SLA)    | 20 dB  | 最大衰减限制       |

**波束增益随角度偏差的变化：**

| 角度偏差$\Delta\psi$ | 衰减$A$    | 线性增益$G_{\text{beam}}$ |
| ---------------------- | ------------ | --------------------------- |
| 0° (主瓣中心)         | 0 dB         | 63.1 (18 dBi)               |
| 15° (半波束宽度)      | 3 dB         | 31.6 (15 dBi)               |
| 30° (波束边缘)        | 12 dB        | 3.98 (6 dBi)                |
| 38.7°                 | 20 dB        | 0.631 (-2 dBi)              |
| > 38.7° (旁瓣)        | 20 dB (饱和) | 0.631 (-2 dBi)              |

##### 2.3.3.5 波束控制模式

智能体可以通过动作控制波束方向，支持两种模式：

**模式 1：Offset 模式（相对偏移）**

波束方向 = 质心方向 + 智能体偏移

$$
\theta = \theta_0 + a_\theta \times \Delta_{\max}, \quad \phi = \phi_0 + a_\phi \times \Delta_{\max}
$$

其中：

- $(\theta_0, \phi_0)$：指向 UE 质心的方向（自动计算）
- $(a_\theta, a_\phi) \in [-1, 1]^2$：智能体动作
- $\Delta_{\max} = 30°$：最大偏移范围

**模式 2：Absolute 模式（绝对角度）**

智能体直接指定波束方向：

$$
\theta = \frac{a_\theta + 1}{2} \times 180°, \quad \phi = a_\phi \times 180°
$$

---

#### 2.3.4 各链路速率计算

所有链路速率计算基于 **Shannon 容量公式**：

$$
R = B \cdot \log_2(1 + \text{SNR}) \quad \text{或} \quad R = B \cdot \log_2(1 + \text{SINR})
$$

##### 2.3.4.1 UE-UAV 下行链路（Edge Link - Downlink）

**场景**：UAV 向其关联的 UE 发送数据

**多址方式**：OFDMA（正交频分多址）

```
        UAV (发射功率 P_tx)
         │
         │  总带宽 B_edge = 30 MHz
         ▼
    ┌────┴────┬────┴────┬────┴────┐
    │ 子带 1  │ 子带 2  │   ...   │  OFDMA: N 个 UE 平分带宽
    │ B/N     │ B/N     │         │
    ▼         ▼         ▼   
   UE_1      UE_2      ...       UE_N
```

**OFDMA 特性分析：**

在 OFDMA 系统中，总功率和带宽平均分配给 $N$ 个 UE：

- 每个 UE 分配带宽：$B_k = B_{\text{edge}} / N$
- 每个 UE 分配功率：$P_k = P_{\text{tx}} / N$

子载波级别的 SINR 计算：

$$
\text{SINR}_k = \frac{P_k \cdot G_k}{\sigma^2 / N + I / N} = \frac{P_{\text{tx}} \cdot G_k}{\sigma^2 + I}
$$

> 由于功率、噪声和干扰都按 $1/N$ 缩放，OFDMA 子载波的 SINR 等于使用全功率时的全频带 SINR。

**下行速率公式：**

$$
R_{\text{DL}}^{(k)} = \frac{B_{\text{edge}}}{N} \cdot \log_2\left(1 + \frac{P_{\text{tx}}^{\text{UAV}} \cdot G_k}{\sigma^2 + I_{\text{total}}^{(k)}}\right)
$$

**参数：**

| 参数         | 符号                           | 值             | 说明          |
| ------------ | ------------------------------ | -------------- | ------------- |
| 边缘链路带宽 | $B_{\text{edge}}$            | 30 MHz         | UE-UAV 总带宽 |
| UAV 发射功率 | $P_{\text{tx}}^{\text{UAV}}$ | 0.8 W          | 总发射功率    |
| 噪声功率     | $\sigma^2$                   | $10^{-13}$ W | AWGN 噪声     |
| 干扰功率     | $I_{\text{total}}$           | 动态计算       | 来自其他 UAV  |

##### 2.3.4.2 UE-UAV 上行链路（Edge Link - Uplink）

**场景**：UE 向其服务 UAV 发送请求消息

**多址方式**：OFDMA（与下行相同频段，但不同时隙或子帧）

**上行速率公式：**

$$
R_{\text{UL}}^{(k)} = \frac{B_{\text{edge}}}{N} \cdot \log_2\left(1 + \frac{P_{\text{tx}}^{\text{UE}} \cdot G_k}{\sigma^2}\right)
$$

> **注**：上行链路暂不考虑同频干扰（保守简化）

**参数：**

| 参数        | 符号                          | 值    | 说明            |
| ----------- | ----------------------------- | ----- | --------------- |
| UE 发射功率 | $P_{\text{tx}}^{\text{UE}}$ | 0.1 W | 远小于 UAV 功率 |

**典型数值示例**（假设 $G = 10^{-8}$, $N = 10$）：

- SNR = $\frac{0.1 \times 10^{-8}}{10^{-13}}$ = 10,000 (40 dB)
- 上行速率 ≈ $\frac{30}{10}$ × $\log_2(10001)$ ≈ 39.9 Mbps

##### 2.3.4.3 UAV-UAV 链路（Inter-UAV Link）

**场景**：UAV 之间的协作通信（内容共享）

**多址方式**：FDM（频分复用）

```
                    UAV_A (协作者)
                   ╱      │      ╲
        ┌─────────╱       │       ╲─────────┐
        │  频段 1         │          频段 2  │
        │  B/2, P/2       │          B/2, P/2│
        ▼                 │                 ▼
     UAV_1              UAV_2             UAV_3
   (请求者)           (其他)            (请求者)
```

当 UAV_A 被 $N_{\text{collab}}$ 个 UAV 选为协作者时：

- 每条链路分配带宽：$B_{\text{link}} = B_{\text{inter}} / N_{\text{collab}}$
- 每条链路分配功率：$P_{\text{link}} = P_{\text{tx}} / N_{\text{collab}}$
- 每条链路噪声功率：$\sigma_{\text{link}}^2 = \sigma^2 / N_{\text{collab}}$（FDM 子频带噪声与带宽成正比）

**UAV-UAV 速率公式：**

$$
R_{\text{UAV-UAV}} = \frac{B_{\text{inter}}}{N_{\text{collab}}} \cdot \log_2\left(1 + \frac{(P_{\text{tx}} / N_{\text{collab}}) \cdot G}{\sigma^2 / N_{\text{collab}}}\right) = \frac{B_{\text{inter}}}{N_{\text{collab}}} \cdot \log_2\left(1 + \frac{P_{\text{tx}} \cdot G}{\sigma^2}\right)
$$

> **注**：由于 FDM 每条链路使用独立子频带，功率和噪声同时缩放 $1/N$，SNR 保持不变。协作者负载仅影响带宽分配，不影响 SNR。

**参数：**

| 参数           | 符号                  | 值     | 说明               |
| -------------- | --------------------- | ------ | ------------------ |
| UAV 间通信带宽 | $B_{\text{inter}}$  | 30 MHz | 专用频段           |
| 协作者负载     | $N_{\text{collab}}$ | 动态   | 被选为协作者的次数 |

**设计考量**：

- 功率和带宽**同时**平分给各链路（总功率限制模型）
- FDM 方式避免了 UAV 间的相互干扰
- 协作者负载越高，单链路速率越低

##### 2.3.4.4 UAV-MBS 链路（Backhaul Link）

**场景**：UAV 与宏基站之间的回程通信

**工作模式**：全双工、点对点

**（A）上行链路（UAV → MBS）**

用于 UAV 向 MBS 发送内容请求：

$$
R_{\text{Backhaul}}^{\text{UL}} = B_{\text{backhaul}} \cdot \log_2\left(1 + \frac{P_{\text{tx}}^{\text{UAV}} \cdot G_{\text{UAV-MBS}}}{\sigma^2}\right)
$$

**（B）下行链路（MBS → UAV）**

用于 MBS 向 UAV 发送请求的内容数据：

$$
R_{\text{Backhaul}}^{\text{DL}} = B_{\text{backhaul}} \cdot \log_2\left(1 + \frac{P_{\text{tx}}^{\text{MBS}} \cdot G_{\text{UAV-MBS}}}{\sigma^2}\right)
$$

**参数：**

| 参数         | 符号                           | 值     | 说明                     |
| ------------ | ------------------------------ | ------ | ------------------------ |
| 回程链路带宽 | $B_{\text{backhaul}}$        | 40 MHz | UAV-MBS 专用             |
| MBS 发射功率 | $P_{\text{tx}}^{\text{MBS}}$ | 20 W   | 宏基站功率（远大于 UAV） |

**非对称速率**：由于 $P_{\text{tx}}^{\text{MBS}} \gg P_{\text{tx}}^{\text{UAV}}$，下行速率通常远高于上行速率。

---

#### 2.3.5 同频干扰模型

在 Edge Link 频段内，所有 UAV 共享相同的 30 MHz 带宽，存在**同频干扰**（Co-Channel Interference）。

##### 2.3.5.1 干扰场景图示

```
                         UAV_j (干扰源)
                          │ 波束方向 →→→
                          │
                     ┌────┴────┐
                     │ 干扰信号 │  干扰功率 = P_tx × G_interference
                     └────┬────┘
                          │
                          ▼
    UAV_i ─────────────► UE_k ◄──── 期望信号
    (服务UAV)            (被干扰)
  
    SINR = 期望信号功率 / (噪声 + 干扰功率)
```

##### 2.3.5.2 干扰功率计算

对于被 UAV $i$ 服务的 UE $k$，来自所有其他 UAV 的干扰功率：

$$
I_{\text{total}}^{(k)} = \sum_{j \neq i} \mathbb{1}_{\{|\mathcal{U}_j| > 0\}} \cdot P_{\text{tx}}^{\text{UAV}} \cdot G_{j \to k}^{\text{(with beam)}}
$$

其中：

- $\mathbb{1}_{\{|\mathcal{U}_j| > 0\}}$：指示函数，UAV $j$ 有关联 UE 时为 1，否则为 0
- $G_{j \to k}^{\text{(with beam)}}$：从干扰 UAV $j$ 到 UE $k$ 的信道增益（考虑 UAV $j$ 的波束方向）

##### 2.3.5.3 干扰功率详细计算

```python
# 伪代码：计算 UE k 受到的总干扰
def calculate_interference(ue_k, serving_uav_i, all_uavs):
    total_interference = 0
  
    for uav_j in all_uavs:
        if uav_j == serving_uav_i:
            continue  # 跳过服务 UAV
  
        if len(uav_j.associated_ues) == 0:
            continue  # 无关联 UE 的 UAV 不发射，不产生干扰
  
        # 计算干扰链路信道增益（考虑 UAV_j 的波束方向）
        G_interference = calculate_channel_gain(
            ue_k.pos, 
            uav_j.pos, 
            beam_direction=uav_j.beam_direction  # 关键：使用干扰UAV的波束方向
        )
  
        # 干扰功率 = 发射功率 × 信道增益
        interference = P_tx_UAV * G_interference
        total_interference += interference
  
    return total_interference
```

##### 2.3.5.4 波束赋形对干扰的抑制作用

由于干扰功率计算考虑了干扰 UAV 的**波束方向**，波束赋形可以有效降低干扰：

| 场景                     | 角度偏差 | 波束增益      | 干扰影响       |
| ------------------------ | -------- | ------------- | -------------- |
| 干扰 UAV 波束指向 UE_k   | ~0°     | 最大 (18 dBi) | 强干扰         |
| 干扰 UAV 波束偏离 30°   | 30°     | -6 dB         | 中等干扰       |
| 干扰 UAV 波束偏离 > 39° | > 39°   | -2 dBi        | 弱干扰（旁瓣） |

> **设计意义**：智能体可以通过调整波束方向，在服务自身 UE 的同时减少对其他 UAV 所服务 UE 的干扰。

##### 2.3.5.5 SINR 计算示例

假设：

- 服务 UAV 到 UE 的信道增益：$G_{\text{signal}} = 10^{-8}$
- 干扰 UAV 数量：3 个
- 每个干扰 UAV 的信道增益：$G_{\text{interf}} = 10^{-10}$（波束偏离导致衰减）

$$
\text{SINR} = \frac{0.8 \times 10^{-8}}{10^{-13} + 3 \times 0.8 \times 10^{-10}} = \frac{8 \times 10^{-9}}{10^{-13} + 2.4 \times 10^{-10}} \approx 33.2 \text{ (15.2 dB)}
$$

---

#### 2.3.6 通信参数汇总

| 参数类别           | 参数名         | 符号                           | 值                       | 单位 |
| ------------------ | -------------- | ------------------------------ | ------------------------ | ---- |
| **发射功率** | UAV 发射功率   | $P_{\text{tx}}^{\text{UAV}}$ | 0.8                      | W    |
|                    | MBS 发射功率   | $P_{\text{tx}}^{\text{MBS}}$ | 20                       | W    |
|                    | UE 发射功率    | $P_{\text{tx}}^{\text{UE}}$  | 0.1                      | W    |
|                    | UAV 接收功率   | $P_{\text{rx}}$              | 0.1                      | W    |
| **带宽**     | Edge Link 带宽 | $B_{\text{edge}}$            | 30                       | MHz  |
|                    | Inter-UAV 带宽 | $B_{\text{inter}}$           | 30                       | MHz  |
|                    | Backhaul 带宽  | $B_{\text{backhaul}}$        | 40                       | MHz  |
| **噪声**     | AWGN 功率      | $\sigma^2$                   | $10^{-13}$             | W    |
| **信道**     | 常数增益因子   | $G_0 \cdot g_0$              | $3.245 \times 10^{-4}$ | -    |
|                    | NLoS 额外损耗  | $L_{\text{NLoS}}$            | 10                       | dB   |
| **波束**     | 最大天线增益   | $G_{\max}$                   | 18                       | dBi  |
|                    | 3dB 波束宽度   | $\psi_{\text{3dB}}$          | 30                       | °   |
|                    | 旁瓣衰减上限   | SLA                            | 20                       | dB   |
| **环境**     | 默认环境类型   | -                              | urban                    | -    |

---

#### 2.3.7 通信模型的设计特点

1. **3D 全向覆盖**：球坐标系波束模型支持对地面和空中 UE 的服务
2. **概率信道模型**：LoS 概率模型反映了真实城市环境的复杂性
3. **智能波束控制**：MARL 智能体可学习优化波束指向，平衡覆盖与干扰
4. **干扰感知**：SINR 计算考虑了多 UAV 同频干扰
5. **差异化链路**：不同类型链路使用独立频段，多址方式各异
6. **非对称回程**：MBS 高功率确保了回程下行的高速率

---

### 2.4 时延模型

#### 2.4.1 内容请求处理流程

```
┌──────────────────────────────────────────────────────────────┐
│                     请求处理流程                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Case 1: 本地缓存命中                                         │
│  ┌────┐  ① 请求  ┌─────┐  ② 数据  ┌────┐                    │
│  │ UE │ ───────► │ UAV │ ───────► │ UE │                    │
│  └────┘  (UL)    └─────┘   (DL)   └────┘                    │
│  Latency = t_req(UE→UAV) + t_data(UAV→UE)                   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Case 2: 协作 UAV 缓存命中                                    │
│  ┌────┐ ① ┌─────┐ ② ┌───────────┐ ③ ┌─────┐ ④ ┌────┐      │
│  │ UE │──►│ UAV │──►│Collaborator│──►│ UAV │──►│ UE │      │
│  └────┘   └─────┘   └───────────┘   └─────┘   └────┘      │
│  Latency = t_req(UE→UAV) + t_req(UAV→Collab)                │
│          + t_data(Collab→UAV) + t_data(UAV→UE) + queue_wait │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Case 3: 通过协作 UAV 从 MBS 获取                             │
│  ┌────┐ ┌─────┐ ┌───────────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐│
│  │ UE │►│ UAV │►│Collaborator│►│ MBS │►│Collab│►│ UAV │►│ UE ││
│  └────┘ └─────┘ └───────────┘ └─────┘ └─────┘ └─────┘ └────┘│
│  Latency = Edge + UAV-UAV + Backhaul + queue_waits          │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Case 4: 直接从 MBS 获取                                      │
│  ┌────┐  ①  ┌─────┐  ②  ┌─────┐  ③  ┌─────┐  ④  ┌────┐     │
│  │ UE │ ──► │ UAV │ ──► │ MBS │ ──► │ UAV │ ──► │ UE │     │
│  └────┘     └─────┘     └─────┘     └─────┘     └────┘     │
│  Latency = Edge + Backhaul + queue_wait                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### 2.4.2 传输时延计算

对于文件 $k$ 的请求，各环节时延为：

**请求消息传输时延：**

$$
t_{\text{req}} = \frac{L_{\text{req}} \times 8}{R_{\text{link}}}
$$

其中 $L_{\text{req}} = 100$ bytes 为请求消息大小。

**数据传输时延：**

$$
t_{\text{data}} = \frac{S_k \times 8}{R_{\text{link}}}
$$

其中 $S_k$ 为文件 $k$ 的大小（bytes）。

#### 2.4.3 排队时延模型详解

系统采用**全双工 + 跨时隙积压**模型，准确模拟现实物理约束下的串行处理：

##### 2.4.3.1 时隙内排队机制

在每个时隙内，不同链路上的传输可以**并行进行**（全双工），但同一链路上新的传输必须**等待前一个传输完成再开始**。系统通过跟踪**积压变量**（backlog）记录链路忙碌程度：

**三种链路类型的排队模式：**

| 链路类型 | 多址模式 | 并行能力 | 积压管理 | 说明 |
|---------|---------|---------|---------|------|
| Edge Link (UE-UAV) | OFDMA | ✓ 同UAV多UE并行 | 单UAV维护_tx/rx_time_ue_uav | 不同UE用不同子载波，同时发送请求和接收数据 |
| Inter-UAV | FDM | ✓ 多请求方并行 | 协作者维护per-requester积压 | 不同请求方用不同频段，协作者的总功率分割 |
| Backhaul (UAV-MBS) | 点对点 | ✗ 点对点排队 | 单UAV维护_tx/rx_time_uav_mbs | 上下行全双工但属于同一UAV，新请求须排队 |

##### 2.4.3.2 跨时隙积压的数学模型

**积压变量定义：**

系统为每个 UAV 维护以下积压状态变量（单位：秒），表示该 UAV 在该链路上的**解决未决传输需求的时间**：
- `_backlog_tx_uav_mbs`: UAV → MBS 上行链路的发送积压（来自前续时隙）
- `_backlog_rx_uav_mbs`: MBS → UAV 下行链路的接收积压
- 类似的 UAV-UAV 链路积压

**本时隙排队等待计算：**

假设在第 $t$ 时隙，某 UAV 有新的传输任务需要在某条链路上处理：

$$
t_{\text{wait}} = \max
\begin{cases}
\text{backlog\_tx} + \text{current\_tx\_time}, & \text{(发送方等待)} \\
\text{backlog\_rx} + \text{current\_rx\_time}  & \text{(接收方等待)}
\end{cases}
$$

其中：
- `backlog_*`: 前续时隙遗留的未完成时间
- `current_*_time`: 本时隙新请求的传输时间累积

**状态更新规则（每时隙末）：**

1. **衰减阶段**（模拟时间前进）：
   ```
   new_backlog = max(old_backlog - tau, 0.0)  # tau=1时隙
   ```

2. **累积阶段**（新请求加入）：
   ```
   new_backlog += current_tx_time  # 或 current_rx_time
   current_tx_time = 0.0  # 重置本时隙累积
   ```

**示例场景**（两个连续时隙）：

```
时隙 t=0:
  - UAV_1 请求 MBS，传输时间 t_tx0 = 0.5s
  - 当前 backlog_tx = 0
  - 等待时间 = max(0 + 0) = 0
  - 完整传输 = 0 + 0.5 = 0.5s > tau=1s，超出本时隙
  - 更新 backlog_tx = 0 + 0.5 - 1.0 = 不会积压（本时隙完成），重置为0

时隙 t=1:
  - UAV_1 又有新请求，传输时间 t_tx1 = 0.8s
  - 上次 backlog_tx = 0（上次完成了）
  - 等待时间 = max(0 + 0) = 0
  - 新 backlog_tx = 0 + 0.8 = 0.8s
```

**高负载场景**（多个请求堆积）：

```
时隙 t=0: 请求1，tx=0.6s
  - backlog = 0, wait = 0
  - 剩余 1.0 - 0.6 = 0.4s （本时隙还有时间）

时隙 t=1: 同时有请求2和请求3
  - 请求2：tx=0.5s
    - backlog=0, wait=0
    - 从0.4s处开始，需要 0.5s，完成于 0.9s，在时隙末
  - 请求3在请求2之前进入队列
    - 等待请求2完成
    - wait = (0 + 0.5) = 0.5s (请求2的传输时间)
    - tx3 = 0.7s
    - 完成于 0.5 + 0.7 = 1.2s > tau，溢出到次时隙
    - new_backlog = 1.2 - 1.0 = 0.2s（待下个时隙继续）

时隙 t=2:
  - 新请求4：tx=0.3s
  - backlog = 0.2s (请求3的遗留)
  - wait = max(0.2 + 0) = 0.2s
  - 队列：[请求3遗留的0.2s] + [请求4的0.3s] = 0.5s，本时隙完成
  - new_backlog = 0
```

##### 2.4.3.3 各链路的排队应用

**Edge Link（OFDMA，不产生排队）：**

- 每个 UE 独用子载波，互不影响
- 同一 UAV 的多请求并行处理
- **无排队等待**（`queue_wait = 0`）

**Inter-UAV Link（FDM，协作者侧排队）：**

- 不同请求方用不同频段（频分复用），无相互干扰
- 同一请求方的多个请求在协作者侧**串行排队**
- 公式：
$$
t_{\text{queue}}^{\text{UAV-UAV}} = \max\left(
\text{my\_backlog\_tx} + \text{my\_tx}, \text{my\_backlog\_rx} + \text{my\_rx}
\right)_{\text{self}} + \max(\cdots)_{\text{at\_collaborator}}
$$

**Backhaul Link（点对点，严格排队）：**

- 上行和下行在独立频段（全双工），但属于同一 UAV
- 同一 UAV 的多个请求必须串行排队
- 公式：
$$
t_{\text{queue}}^{\text{MBS}} = \max\left(
\text{backlog\_tx\_mbs} + \text{tx\_time\_mbs}, \text{backlog\_rx\_mbs} + \text{rx\_time\_mbs}
\right)
$$

##### 2.4.3.4 最终 UE 延迟的组成

对于 Case 2（协作UAV缓存命中）的完整延迟：

$$
\text{Latency}_{UE} = \underbrace{t_{\text{req}}(UE \to UAV) + t_{\text{data}}(UAV \to UE)}_{\text{Edge Link}}
+ \underbrace{t_{\text{req}}(UAV \to Collab) + t_{\text{data}}(Collab \to UAV)}_{\text{Inter-UAV Link}}
+ \underbrace{t_{\text{queue}}^{\text{UAV-UAV}}}_{\text{积压等待}}
$$

其中 `queue_wait` 包含了前续时隙遗留的积压和本时隙新请求的排队。

##### 2.4.3.5 实现细节与代码映射

代码实现在 [uavs.py](environment/uavs.py#L380-L490) 中：

```python
# 本时隙积压状态（秒）
self._backlog_tx_uav_mbs: float  # 前一时隙遗留的 MBS 上行发送
self._backlog_rx_uav_mbs: float  # 前一时隙遗留的 MBS 下行接收

# 本时隙累积的传输时间（用于新请求排队）
self._tx_time_uav_mbs: float  # 本时隙已累积的 MBS 上行传输时间
self._rx_time_uav_mbs: float  # 本时隙已累积的 MBS 下行接收时间

# 线路排队等待（包含积压）
queue_wait_uav_mbs = max(
    self._backlog_tx_uav_mbs + self._tx_time_uav_mbs,
    self._backlog_rx_uav_mbs + self._rx_time_uav_mbs
)
```

**时隙末更新**（[uavs.py#L580](environment/uavs.py#L580)）：

```python
# 积压衰减：已过去 1 时隙
self._backlog_tx_uav_mbs = max(self._backlog_tx_uav_mbs - 1.0, 0.0)
self._backlog_rx_uav_mbs = max(self._backlog_rx_uav_mbs - 1.0, 0.0)

# 如果本时隙传输未完成，转入下一时隙的积压
if self._tx_time_uav_mbs > 1.0:
    self._backlog_tx_uav_mbs += self._tx_time_uav_mbs - 1.0
if self._rx_time_uav_mbs > 1.0:
    self._backlog_rx_uav_mbs += self._rx_time_uav_mbs - 1.0

# 重置本时隙累积
self._tx_time_uav_mbs = 0.0
self._rx_time_uav_mbs = 0.0
```

##### 2.4.3.6 与真实系统的对应关系

- **OFDMA 并行处理**：对应实际蜂窝系统中多用户共享 RB（资源块）的调度
- **FDM 频分复用**：对应 UAV 间的专用频段分配，避免干扰
- **跨时隙积压**：对应实际网络中的缓冲队列行为 —— 当传输需求超过时隙容量时，剩余请求延到次时隙或更晚处理

---

#### 2.4.4 能耗与时延的权衡

系统在优化过程中需要平衡时延和能耗：

- **短时延链路**：Edge（OFDMA）> Inter-UAV（FDM）> Backhaul（点对点限流）
- **低能耗**：协作传输（利用缓存）< 直接 MBS 获取（回程功率高，距离远）
- **MARL 学习目标**：通过合理的 UAV 部署、波束控制、协作选择，找到时延/能耗最优平衡

$$
t_{\text{queue}} = \max(t_{\text{backlog}}^{\text{tx}} + t_{\text{current}}^{\text{tx}}, t_{\text{backlog}}^{\text{rx}} + t_{\text{current}}^{\text{rx}})
$$

**跨时隙积压更新：**

$$
t_{\text{backlog}}^{\text{next}} = \max(0, t_{\text{total}} - \tau)
$$

其中 $\tau = 1$s 为时隙长度，超出时隙的传输任务积压到下一时隙。

#### 2.4.4 未服务惩罚

对于未被任何 UAV 覆盖的 UE，其时延设为惩罚值：

$$
t_{\text{unserved}} = 60 \text{s}
$$

---

### 2.5 能耗模型

UAV 的能耗分为**飞行能耗**和**通信能耗**两部分。

#### 2.5.1 飞行能耗

$$
E_{\text{fly}} = P_{\text{move}} \cdot t_{\text{move}} + P_{\text{hover}} \cdot t_{\text{hover}}
$$

其中：

- $P_{\text{move}} = 60$ W：移动功率
- $P_{\text{hover}} = 40$ W：悬停功率
- $t_{\text{move}} = \frac{d_{\text{moved}}}{v^{\text{UAV}}}$：移动时间
- $t_{\text{hover}} = \tau - t_{\text{move}}$：悬停时间

#### 2.5.2 通信能耗

$$
E_{\text{comm}} = P_{\text{tx}} \cdot t_{\text{tx}}^{\text{total}} + P_{\text{rx}} \cdot t_{\text{rx}}^{\text{total}}
$$

其中：

- $P_{\text{tx}} = 0.8$ W：发射功率
- $P_{\text{rx}} = 0.1$ W：接收功率

**通信时间的计算特点：**

| 链路类型          | 多址方式 | 时间计算             |
| ----------------- | -------- | -------------------- |
| UE-UAV            | OFDMA    | 取最大值（并行传输） |
| UAV-UAV（请求方） | 点对点   | 累加（串行）         |
| UAV-UAV（协作者） | FDM      | 取最大值（并行传输） |
| UAV-MBS           | 点对点   | 累加（串行）         |

**能耗计算示例：**

$$
t_{\text{tx}}^{\text{total}} = t_{\text{tx}}^{\text{UE-UAV}} + t_{\text{tx}}^{\text{UAV-UAV}} + t_{\text{tx}}^{\text{UAV-MBS}}
$$

（不同频段使用独立射频前端，时间累加）

#### 2.5.3 总能耗

$$
E_{\text{total}} = E_{\text{fly}} + E_{\text{comm}}
$$

---

### 2.6 缓存模型

#### 2.6.1 内容请求模型

UE 的内容请求遵循 **Zipf 分布**：

$$
P(k) = \frac{k^{-\beta}}{\sum_{i=1}^{K} i^{-\beta}}
$$

其中：

- $K = 20$：内容文件总数
- $\beta = 0.6$：Zipf 参数（控制热门程度集中度）

#### 2.6.2 GDSF 缓存策略

系统采用 **GDSF（Greedy-Dual-Size-Frequency）** 缓存策略：

**优先级分数计算：**

$$
\text{Priority}_k = \frac{\text{EMA}_k}{S_k}
$$

其中 EMA（指数移动平均）用于平滑请求频率：

$$
\text{EMA}_k(t) = \alpha \cdot f_k(t) + (1-\alpha) \cdot \text{EMA}_k(t-1)
$$

- $\alpha = 0.5$：平滑因子
- $f_k(t)$：当前时隙文件 $k$ 的请求次数

**缓存更新周期：** 每 $T_{\text{cache}} = 10$ 时隙更新一次

**缓存选择算法：**

```python
sorted_files = sort_by_priority_descending(all_files)
cache = []
used_space = 0
for file in sorted_files:
    if used_space + file.size <= storage_capacity:
        cache.append(file)
        used_space += file.size
```

#### 2.6.3 协作缓存

当本地缓存未命中时，UAV 可从协作者获取内容。

**协作者选择策略：**

1. 计算缺失文件集合：$\mathcal{F}_{\text{miss}} = \mathcal{F}_{\text{req}} \setminus \mathcal{F}_{\text{cache}}$
2. 对每个邻居计算缓存重叠：$\text{overlap}_j = |\mathcal{F}_{\text{miss}} \cap \mathcal{F}_{\text{cache}}^{(j)}|$
3. 选择重叠最大的邻居作为协作者（相同则选最近）

---

### 2.7 奖励设计

#### 2.7.1 多目标奖励函数

$$
r = \alpha_3 \cdot r_{\text{fairness}} + \alpha_{\text{rate}} \cdot r_{\text{rate}} - \alpha_1 \cdot r_{\text{latency}} - \alpha_2 \cdot r_{\text{energy}}
$$

**权重设置：**

| 权重                     | 值  | 说明           |
| ------------------------ | --- | -------------- |
| $\alpha_1$             | 1.0 | 时延惩罚权重   |
| $\alpha_2$             | 1.0 | 能耗惩罚权重   |
| $\alpha_3$             | 1.0 | 公平性奖励权重 |
| $\alpha_{\text{rate}}$ | 1.0 | 吞吐量奖励权重 |

#### 2.7.2 奖励缩放与对数压缩

为平衡不同量级的指标，避免在训练初期产生巨大的梯度，采用**多层缩放机制**：

**第一层：静态比例缩放**

根据环境规模预设常量缩放因子，将各项指标归一化到合理范围：

| 指标 | 缩放参数 | 计算公式 | 默认值 | 说明 |
|------|---------|---------|--------|------|
| 时延 | $B_{\text{lat}}$ | $\text{NUM\_UES} \times \tau \times 6$ | 600 | 使 scaled_latency ≈ 0.1~10 |
| 能耗 | $B_{\text{nrg}}$ | $\text{NUM\_UAVS} \times P_{\text{hover}} \times \tau \times 3$ | 1200 | 使 scaled_energy ≈ 0.1~10 |
| 吞吐量 | $B_{\text{rate}}$ | 固定值 | $5 \times 10^7$ | 使 scaled_rate ≈ 0.5~10 |

$$\hat{x} = \frac{x}{B_x + \epsilon}$$

**第二层：对数压缩**

在缩放后应用对数函数，平滑指标边界，避免极端值导致梯度爆炸：

$$r_{\text{lat}} = \alpha_1 \cdot \log(1 + \hat{\text{latency}})$$
$$r_{\text{nrg}} = \alpha_2 \cdot \log(1 + \hat{\text{energy}})$$
$$r_{\text{rate}} = \alpha_{\text{rate}} \cdot \log(1 + \hat{\text{rate}})$$

**第三层：组合与全局缩放**

所有分项奖励组合后，再乘以全局缩放因子以控制奖励量级：

$$r_{\text{combined}} = r_{\text{fairness}} + r_{\text{rate}} - r_{\text{lat}} - r_{\text{nrg}}$$

$$r_{\text{final}} = r_{\text{combined}} \times \text{REWARD\_SCALING\_FACTOR}$$

其中 $\text{REWARD\_SCALING\_FACTOR} = 0.12$ 用于将最终奖励约束在 $[-1, 1]$ 范围内。

**设计意义**：这种多层缩放方案保证了：
- 不同量纲指标相互平衡（第一层）
- 梯度平滑且数值稳定（第二层）
- 奖励幅度适合策略学习（第三层）

#### 2.7.3 公平性指标（Jain's Fairness Index）

$$
\text{JFI} = \frac{\left(\sum_{m=1}^{M} c_m\right)^2}{M \cdot \sum_{m=1}^{M} c_m^2}
$$

其中 $c_m$ 是 UE $m$ 的滑动窗口服务覆盖率（最近 100 步内被成功服务的比例）。

**JFI 映射到奖励**

JFI 取值范围 $[0, 1]$，通过仿射变换和裁剪映射到奖励空间：

$$r_{\text{fairness}} = \alpha_3 \cdot \text{clip}\left((JFI - b) \times s, r_{\min}, r_{\max}\right)$$

其中各参数的具体值和含义为：

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 基准点 | $b$ | 0.6 | JFI 值的"中点"；当 JFI=0.6 时，奖励为 0 |
| 斜率 | $s$ | 5.0 | 放大系数；使公平性偏差 ±0.1 对应 ±0.5 奖励 |
| 奖励下界 | $r_{\min}$ | -2.0 | 最低公平性惩罚上限 |
| 奖励上界 | $r_{\max}$ | 2.0 | 最高公平性奖励上限 |

**映射曲线示例**

| JFI 值 | $(JFI - 0.6) \times 5$ | 最终奖励 (clip) | 说明 |
|-------|----------------------|-----------------|------|
| 0.2 | -2.0 | -2.0 | 极度不公平，满惩罚 |
| 0.5 | -0.5 | -0.5 | 公平性较差 |
| 0.6 | 0.0 | 0.0 | 基准点，无奖惩 |
| 0.7 | 0.5 | 0.5 | 公平性良好 |
| 1.0 | 2.0 | 2.0 | 完全公平，满奖励 |

这种设计确保了公平性指标与奖励的平衡性：既不会因为单次偏差过度惩罚，也能鼓励系统趋向更公平的服务分配。

#### 2.7.4 碰撞安全约束与惩罚

系统采用**分层惩罚机制**处理 UAV 间的碰撞风险，触发条件与惩罚值如下：

| 违规类型 | 触发条件 | 惩罚计算 | 默认值 |
|---------|---------|--------|--------|
| 危险靠近 | $5\text{m} < d_{\min} < 50\text{m}$ | $P_{\text{unsafe}} \times \text{ratio}$（动态线性插值） | $P_{\text{unsafe}} = 4.0$ |
| 碰撞失效 | $d_{\min} \leq 5\text{m}$ | 固定惩罚 + 标记失效 | $P_{\text{collision}} = 10.0$ |
| 边界违规 | 越界动作 | 固定惩罚 | $P_{\text{boundary}} = 2.0$ |

**危险靠近惩罚的详细计算**

当两架 UAV 的同步最小距离 $d_{\min}$ 落在 $(5\text{m}, 50\text{m})$ 区间时，会受到**线性插值的动态惩罚**：

$$\text{ratio} = \frac{50 - d_{\min}}{50 - 5} = \frac{50 - d_{\min}}{45}$$

$$r_{\text{unsafe}} = -P_{\text{unsafe}} \times \text{clip}(\text{ratio}, 0, 1)$$

- 当 $d_{\min} = 45\text{m}$ 时，ratio = 0.111，惩罚 ≈ -0.44
- 当 $d_{\min} = 25\text{m}$ 时，ratio = 0.556，惩罚 ≈ -2.22
- 当 $d_{\min} = 5\text{m}$ 时，ratio = 1.0，触发碰撞失效

**碰撞失效的后续影响**

一旦两架 UAV 发生碰撞（$d_{\min} \leq 5\text{m}$）：
1. 两者立即标记为失效（`failed=True`）
2. 本回合剩余时隙内，失效 UAV 的动作被忽略
3. 每个失效 UAV 额外扣除 10.0 的固定惩罚
4. 失效 UAV 的回程链路积压被清空

**最终奖励计算**

所有惩罚项在时隙末尾累加应用于 UAV 的回合奖励：

$$r_{\text{final}} = (r_{\text{shared}} - \sum \text{penalties}) \times 0.12$$

其中惩罚项包括：`proximity_penalty`（动态）、`collision_failure_penalty`（10.0）、`boundary_penalty`（2.0）。

---

#### 奖励设计参数速查表

为方便调参，将所有与奖励相关的配置集中在一个表中：

| 参数类别 | 参数名 | 值 | 类型 | 说明 |
|--------|-------|-----|------|------|
| **权重** | $\alpha_1$ (时延) | 1.0 | 无量纲 | 时延惩罚权重（固定） |
| | $\alpha_2$ (能耗) | 1.0 | 无量纲 | 能耗惩罚权重（固定） |
| | $\alpha_3$ (公平性) | 1.0 | 无量纲 | 公平性奖励权重（固定） |
| | $\alpha_{\text{rate}}$ (吞吐量) | 1.0 | 无量纲 | 吞吐量奖励权重（固定） |
| **缩放** | $B_{\text{lat}}$ (时延SCALE) | 600 | 秒 | 近似值 = `NUM_UES × τ × 6` |
| | $B_{\text{nrg}}$ (能耗SCALE) | 1200 | 焦耳 | 近似值 = `NUM_UAVS × P_hover × τ × 3` |
| | $B_{\text{rate}}$ (速率SCALE) | $5 \times 10^7$ | bps | 固定值 |
| | REWARD_SCALING_FACTOR | 0.12 | 无量纲 | 全局奖励缩放因子 |
| **JFI** | JFI 基准点 | 0.6 | 无量纲 | 公平性中点 |
| | JFI 斜率 | 5.0 | 无量纲 | 公平性灵敏度 |
| | JFI 奖励下界 | -2.0 | 无量纲 | 最低公平性惩罚 |
| | JFI 奖励上界 | 2.0 | 无量纲 | 最高公平性奖励 |
| **失效UE** | 未服务时延惩罚 | 60.0 | 秒 | 未被覆盖 UE 的时延值 |

**调参建议**：
- 权重（$\alpha_*$）通常保持为 1.0，通过 SCALE 参数实现目标间的平衡
- SCALE 参数应根据环境规模（NUM_UAVS、NUM_UES）重新计算；公式见本表
- REWARD_SCALING_FACTOR 影响收敛速度，通常在 0.05~0.20 之间调整

---

## 3. 算法设计

### 3.1 观测空间

每个 UAV 智能体的观测向量维度由配置项决定：

$$
\text{OBS\_DIM\_SINGLE} = \text{OWN\_STATE\_DIM} + (N_{\text{nbr}} \times \text{NEIGHBOR\_STATE\_DIM}) + (N_{\text{UE}} \times \text{UE\_STATE\_DIM}) + 2
$$

其中：

- $\text{OWN\_STATE\_DIM} = 24$（位置(3) + 缓存(20) + 活跃标记(1)）
- $N_{\text{nbr}} = \text{MAX\_UAV\_NEIGHBORS}$（邻居 UAV 上限，当前配置为 4）
- $\text{NEIGHBOR\_STATE\_DIM} = 26$（相对位置(3) + 缓存位图(20) + 即时帮助能力(1) + 缓存互补性(1) + 活跃标记(1)）
- $N_{\text{UE}} = \text{MAX\_ASSOCIATED\_UES}$（关联 UE 上限，当前配置为 50）
- $\text{UE\_STATE\_DIM} = 5$（相对位置(3) + 归一化文件ID(1) + 缓存命中标志(1)）
- 额外 2 维：邻居数量原始计数(1) + 关联UE数量原始计数(1)

因此在默认配置（NUM_UAVS=10, NUM_UES=100）下：

$$
\text{OBS\_DIM\_SINGLE} = 24 + 4 \times 26 + 50 \times 5 + 2 = 380
$$

#### 3.1.1 观测结构

```
观测向量 (默认 380 维)
├── 自身状态 (24 维)
│   ├── 归一化位置 [x/X_max, y/Y_max, z/Z_max]  ... 3 维
│   ├── 缓存位图 [cache_0, cache_1, ..., cache_19] ... 20 维
│   └── 活跃标记 (0/1) ... 1 维
│
├── 邻居状态 (4 × 26 = 104 维)
│   └── 每个邻居 (26 维):
│       ├── 相对位置 [Δx, Δy, Δz] / R_sense  ... 3 维
│       ├── 缓存位图 [cache_0, cache_1, ..., cache_19] ... 20 维
│       ├── 即时帮助能力 (0~1)  ... 1 维
│       ├── 缓存互补性 (0~1)  ... 1 维
│       └── 活跃标记 (0/1)  ... 1 维
│
├── 关联 UE 状态 (50 × 5 = 250 维)
│   └── 每个 UE (5 维):
│       ├── 相对位置 [Δx, Δy, Δz] / R_coverage  ... 3 维
│       ├── 归一化请求文件ID  ... 1 维
│       └── 本地缓存命中标志 (0/1)  ... 1 维
│
└── 实体计数记录 (2 维)
    ├── 邻居数量 (原始计数值，范围 [0, MAX_UAV_NEIGHBORS]，默认0~4) ... 1 维
    └── 关联UE数量 (原始计数值，范围 [0, MAX_ASSOCIATED_UES]，默认0~50) ... 1 维
```

**说明**：计数字段采用原始整数值（未归一化）。在注意力机制中，这两个字段用于**动态生成 Mask**：将计数值转换为 boolean mask，使注意力机制只关注真实数据而忽略填充。在当前无注意力实现中，MADDPG 仍通过 `MeanPoolingEncoder` 利用这些计数做均值池化；MAPPO 无注意力分支则直接保留这些原始计数字段作为输入特征的一部分，不再显式做 entity-level 均值池化。

#### 3.1.2 观测特征详解

**（1）自身状态**

| 特征       | 维度 | 值域   | 说明                                                                           |
| ---------- | ---- | ------ | ------------------------------------------------------------------------------ |
| 归一化位置 | 3    | [0, 1] | $(x/1000, y/1000, z/600)$，Z轴上限为 $\text{UE\_MAX\_ALT}=600\text{m}$（包含空中UE的最大高度） |
| 缓存位图   | 20   | {0, 1} | 第$k$ 位为 1 表示缓存了文件 $k$                                            |
| 活跃标记   | 1    | {0, 1} | 当前无人机是否存活/活跃（1=活跃，0=失效）                                   |

**（2）邻居状态**（最多 4 个最近邻居）

| 特征         | 维度 | 范围    | 计算方式                                                                           |
| ------------ | ---- | ------- | ---------------------------------------------------------------------------------- |
| 相对位置     | 3    | [-1, 1] | $(\vec{p}_{\text{neighbor}} - \vec{p}_{\text{self}}) / R^{\text{sense}}$         |
| 缓存位图     | 20   | {0, 1}  | 邻居的缓存状态位图                                                                 |
| 即时帮助能力 | 1    | [0, 1]  | 该邻居能帮助解决本地无法命中请求的比例                                             |
| 缓存互补性   | 1    | [0, 1]  | 自己缺少的全体文件中，该邻居能提供的比例                                           |
| 活跃标记     | 1    | {0, 1}  | 当前邻居是否存活/活跃                                                              |

**（3）关联 UE 状态**（最多 $N_{\text{UE}}=\text{MAX\_ASSOCIATED\_UES}$ 个最近 UE；默认配置为 50）

| 特征          | 维度 | 范围   | 说明                                                 |
| ------------- | ---- | ------ | ---------------------------------------------------- |
| 相对位置      | 3    | ~      | $(\vec{p}_{\text{UE}} - \vec{p}_{\text{UAV}}) / R$ |
| 归一化文件 ID | 1    | [0, 1] | $\text{req\_id} / K$                               |
| 缓存命中标志  | 1    | {0, 1} | 本 UAV 是否缓存该请求文件                            |

**（4）实体计数（2 维）**
分别提供有效关联到的观测邻居及关联UE数量（原始计数，未在环境侧归一化）。注意力分支直接据此构建 mask；无注意力的 MADDPG 均值池化编码器内部会再按最大值归一化这两个计数特征。

### 3.2 动作空间

每个 UAV 智能体输出 **5 维连续动作**：

```
动作向量 (5 维)
├── 移动动作 (3 维)
│   ├── dx ∈ [-1, 1]  ... X 方向归一化位移
│   ├── dy ∈ [-1, 1]  ... Y 方向归一化位移
│   └── dz ∈ [-1, 1]  ... Z 方向归一化位移
│
└── 波束控制动作 (2 维)
    ├── beam_θ ∈ [-1, 1]  ... 俯仰角控制
    └── beam_φ ∈ [-1, 1]  ... 方位角控制
```

#### 3.2.1 移动动作解释

**位移向量计算：**

$$
\vec{\Delta p} = \frac{\vec{a}_{\text{move}}}{\max(\|\vec{a}_{\text{move}}\|, 1)} \times v^{\text{UAV}} \times \tau
$$

即动作向量的方向决定移动方向，模长（裁剪到最大为 1）决定移动距离占最大距离的比例。

**示例：**

- $\vec{a} = [1, 0, 0]$：最大速度向 X+ 移动
- $\vec{a} = [0.5, 0.5, 0]$：以约 71% 最大速度沿对角线移动
- $\vec{a} = [0, 0, 0]$：悬停

#### 3.2.2 波束控制动作解释

**Absolute 模式（默认，当前配置 `BEAM_CONTROL_MODE="absolute"`）：**

$$
\theta = (a_\theta + 1) / 2 \times 180°
$$

$$
\phi = a_\phi \times 180°
$$

其中 $\theta \in [0, 180°]$，$\phi \in [-180°, 180°]$。

**Offset 模式（可选，`BEAM_CONTROL_MODE="offset"`）：**

$$
\theta = \theta_0 + a_\theta \times 30°
$$

$$
\phi = \phi_0 + a_\phi \times 30°
$$

其中 $(\theta_0, \phi_0)$ 是指向关联 UE 质心的方向，$30°$ 来自配置 `BEAM_OFFSET_RANGE=30.0`。

### 3.3 支持的 MARL 算法

本项目采用 CTDE（集中式训练，分布式执行）架构，支持多种前沿的多智能体强化学习算法，下面重点介绍其核心算法的实现特点：

#### 3.3.1 MADDPG (多智能体深度确定性策略梯度)
作为本项目的核心 Off-policy 算法基线，其经历了深度定制：
1. **注意力协同架构（Attention Mechanism）**：通过可配置的 `USE_ATTENTION`，MADDPG 和 MAPPO 现已支持基于 Cross-Attention 的状态编码器，以处理可变长度的 UE 列表和邻居状态。
2. **共享编码器与置换不变性**：在 Attention 模式下，多个 Agent 的 Critic 网络共享底层的注意力特征提取器以提升样本效率。其 `AgentPoolingAttention` 模块实现了真正的置换不变（Permutation-Invariant）聚合，使 Critic 能稳定且可扩展地评估全局状态。
3. **掩码控制（Active / Bootstrap Masking）**：深度集成了防失效智能体干扰机制，屏蔽发生崩溃出界截断的 Agent，并在目标 Critic 评估时使用精准的 next-step Bootstrap 掩码。

#### 3.3.2 MAPPO (多智能体近端策略优化)
作为本项目的核心 On-policy 算法基线，针对复杂的 UAV 微调场景解决了多个关键痛点：
1. **有界动作空间与 Jacobian 校正**：摒弃了易出现越界截断的传统 Normal 采样裁剪方案，环境直接映射 $[-1, 1]$ 有界动作，而在算法层采用 `tanh-squash` 机制并附加 Jacobian 行列式校正。保证了动作采样不越界且 Log Probability 计算的绝对准确。
2. **Rollout 级别的 Advantage 归一化**：不同于在每个 mini-batch 内重复归一化（易引入高方差），系统利用 Rollout Buffer 的全量视角进行全局 Advantage 归一化，且仅针对 Active 的智能体生效，极大提升了训练稳定性。
3. **critic 粒度已分化**：当前 MAPPO 的两条 critic 分支不再完全同构。non-attention critic 直接接收 `flat share_obs` 并输出单个标量 `V(s)`，在 rollout 阶段广播为每个 agent 的 baseline；attention critic 则已升级为 per-agent centralized critic，路径为 `share_obs -> per-agent AttentionEncoder -> concat(team_context) -> e_i 条件化调制 -> per-agent values`。因此，attention 分支已经部分缓解了共享团队基线过粗的问题，而 non-attention 分支仍保留较粗粒度的团队 baseline。
4. **双分支实现（Attention / Non-Attention）**：当 `USE_ATTENTION=True` 时，MAPPO 使用 `AttentionActorNetwork + AttentionCriticNetwork`，其中 attention critic 内部带有 team-context conditioner；当 `USE_ATTENTION=False` 时，普通 actor 直接使用 `raw obs`，普通 critic 直接使用 `flat share_obs`。两条分支共享统一 critic 输入契约：`share_obs` 的形状为 `[batch, num_agents * obs_dim]`，但内部处理路径不同。
5. **工程鲁棒性与训练统计**：MAPPO 训练路径保留 checkpoint metadata 校验与原子回滚机制，避免 attention/non-attention 配置错配导致的部分加载污染。训练主日志与调试日志拆分为两个 JSONL 文件：环境表现指标写入 `log_data_<timestamp>.json`，训练诊断指标按较低频率写入 `debug_data_<timestamp>.json`。

#### 3.3.3 其他算法模型与通用网络配置
| 算法             | 类型       | Actor  | Critic    | 核心特点                 |
| ---------------- | ---------- | ------ | --------- | -------------------- |
| **MATD3**  | Off-policy | 分布式 | 双 Critic | 引入双网络与策略延迟更新减少 Q 值过估计 |
| **MASAC**  | Off-policy | 分布式 | 双 Critic | 基于最大熵探索优化，提升鲁棒性 |
| **Random** | -          | 随机   | -         | 用于通信与系统仿真的基线性能对比 |

##### 网络与超参数基线配置
- **MLP 隐藏层维度**：768 (Shared block configuration)
- **Actor 网络结构**：
    - MADDPG/MATD3: `obs_dim → 768 → 768 → action_dim` (确定性策略，直接 tanh 输出)
    - MASAC: `obs_dim → 768 → 768 → (mean, log_std)`（随机策略，重参数采样后 tanh）
  - MAPPO: `obs_dim → 768 → 768 → action_dim (均值) + log_std` (正态分布与 tanh-squash 机制)
    - *注*：MADDPG 在无注意力配置下仍使用 `MeanPoolingEncoder` 缓解 padding 对输入语义的污染；MAPPO 无注意力配置已改为直接使用原始 `obs/share_obs`。
- **Critic 网络结构**：
    - MATD3/MASAC: `(num_agents*obs_dim + num_agents*action_dim) → 768 → 768 → 1`（双 Critic 版本，无残差）
    - MAPPO:
      - non-attention: `share_obs(flattened joint obs) → scalar value head → 1`
      - attention: `share_obs(flattened joint obs) → reshape([batch, N, obs_dim]) → per-agent AttentionEncoder → concat(team_context) → e_i-conditioned modulation → shared scalar value head → [N]`
    - MADDPG: `joint_obs + joint_action → (MeanPoolingEncoder 或共享 AttentionEncoder + AgentPoolingAttention) → 768 残差 MLP → 1`
- **优化器与学习率**：Actor $LR = 1 \times 10^{-4}$，Critic $LR = 2 \times 10^{-4}$，AdamW

| 通用超参数         | 值 | 说明 |
|------------------|----|------|
| 分配折现 $\gamma$ | 0.99 | 价值未来折扣率 |
| 软更新网络 $\tau$ | 0.001 | 目标网络的平滑更新率 |
| 经验回放与批量 | Buffer: $6\times10^5$, Batch: 1024 | |
| MAPPO Epoch | 10 | PPO阶段迭代轮数 |

---

## 4. 仿真流程

### 每时隙执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    时隙开始 (t → t+1)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 环境准备 (_prepare_for_next_step)                 │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ● UE 生成内容请求 (Zipf 分布)                        │   │
│  │ ● UE-UAV 关联 (3D 球形覆盖 + 最近选择)               │   │
│  │ ● 设置 UAV 邻居列表                                  │   │
│  │ ● 选择协作 UAV                                       │   │
│  │ ● 计算通信速率                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. 获取观测 (_get_obs)                               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ● 构建每个 UAV 的 OBS_DIM_SINGLE 维观测向量          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. 智能体决策 (model.select_actions)                 │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ● Actor 网络输出 5 维动作                            │   │
│  │ ● 添加探索噪声 (训练时)                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. 执行动作 (env.step)                               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ● 应用波束控制动作                                   │   │
│  │ ● 计算同频干扰                                       │   │
│  │ ● 处理内容请求 (计算时延)                            │   │
│  │ ● 更新缓存                                           │   │
│  │ ● 执行 UAV 移动 (碰撞避免)                           │   │
│  │ ● 计算能耗                                           │   │
│  │ ● 计算奖励                                           │   │
│  │ ● 更新 UE 位置                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 5. 学习更新 (model.update)                           │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ● 存储经验到 Buffer                                  │   │
│  │ ● 采样 Batch 更新网络                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 训练日志与离线绘图

当前仓库的日志与绘图链路已经过一次明确收缩：环境表现指标保真记录，展示层只做 `raw + EMA`，不再混入误差带或置信区间。

#### 4.1.1 日志文件

训练阶段会在 `train_logs/` 目录下生成三类文件：

| 文件 | 频率 | 内容 |
|------|------|------|
| `log_data_<timestamp>.json` | 每个 `episode/update` 一条 | 环境表现指标：`reward / latency / energy / fairness / rate / collisions / boundaries / time` |
| `debug_data_<timestamp>.json` | 每 `LOG_FREQ` 个 `episode/update` 一条 | 训练诊断指标：如 `actor_loss / critic_loss / entropy / ratio_mean / clip_fraction / log_std_mean / action_std` |
| `config_<timestamp>.json` | 每次训练一次 | 当次训练使用的配置快照 |

其中：

- `MAPPO` 使用 `update` 作为进度单位；
- `MADDPG / MATD3 / MASAC / Random` 使用 `episode` 作为进度单位；
- 自动绘图与 `state_images` 生成频率由 `PLOT_FREQ=50` 控制，不会在每个 step 都触发。

#### 4.1.2 在线绘图与离线绘图

当前绘图逻辑由 `utils/plot_logs.py` 实现，核心规则如下：

1. 单指标图均采用“浅色原始细线 + 深色 EMA 趋势线”。
2. 不再绘制误差带、标准差阴影或置信区间。
3. 双 y 轴对比图仍保留，包括：
   - Reward + Fairness
   - Latency + Energy
   - Rate + Fairness

离线绘制单个日志文件：

```bash
python plot_existing_logs.py train_logs/log_data_YYYY-MM-DD_HH-MM-SS.json --output_dir output_dir --smoothing 0.9
```

离线绘制多算法对比图：

```bash
python plot_comparison.py \
  --files train_logs/log_data_a.json train_logs/log_data_b.json \
  --labels MAPPO MADDPG \
  --output comparison_reward.png \
  --metric reward \
  --smoothing 0.9
```

> 注意：多算法对比要求所有输入文件使用相同的 x 轴类型，不能混用 `episode` 与 `update`。

---

## 附录：快速启动

### 训练

```bash
python main.py train --num_episodes 5000 --gpu_id 0
```

### 测试

```bash
python main.py test --num_episodes 100 --model_path <path> --config_path <path>
```

### 可视化

```bash
python visualize.py
```

### 离线绘图

```bash
python plot_existing_logs.py train_logs/log_data_YYYY-MM-DD_HH-MM-SS.json --output_dir output_dir --smoothing 0.9
```

### 多算法对比绘图

```bash
python plot_comparison.py \
  --files train_logs/log_data_a.json train_logs/log_data_b.json \
  --labels MAPPO MADDPG \
  --output comparison_reward.png \
  --metric reward \
  --smoothing 0.9
```

---

*文档更新：2026年4月4日（已按当前代码实现重新校准）*
