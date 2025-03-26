# DiffusionForTimeSeries

Diffusion Timeseries 是一个基于扩散（Diffusion）思想的时序预测模型框架，在一些时间序列数据（例如金融数据）预测任务上表现优秀。项目基于 PyTorch 实现，通过配置 `config` 字典提供灵活的参数设置，方便用户调整模型参数进行实验。

## 目录 Content
[TOC]

## Features

模型整体采用了扩散模型（diffusion model）的基本思想，通过逐步添加噪声到原始数据，再利用去噪网络（Denoiser）进行重构，进而达到预测未来序列值的目的。另外，框架还包括：

+ 动态噪声调度（Dynamic Beta Schedule）
+ 复合泊松跳跃噪声模块/自适应跳跃（Compound Poisson/Adaptive Jump Module）
+ 多尺度 Transformer 条件编码器（MultiScale Condition Encoder）
+ 时间嵌入（Sinusoidal Embedding & TimeEmbedding）

在训练程中，模型使用的是 **稳健异方差损失函数（Robust Heteroscedastic Loss）** 来同时拟合目标序列和噪声风险，最后通过 **DDIM（Denoising Diffusion Implicit Models）** 采样生成预测样本，并在测试集上进行评估。

## 项目结构

```plaintext
DiffusionForTimeSeries/
├── All_methods/     # 工具
│   ├── __pycache__
│   ├── _init_.py   
│   └── assess.py    # 计算结果评估工具
├── __init__.py
├── cli.py           # 命令行工具
├── denoiser.py      # 去噪网络，直接预测 x0 与风险指标
├── diffusion.py     # 扩散模块、动态/静态 beta 生成函数
├── embeddings.py    # 时间嵌入与 sinusoidal 嵌入函数
├── encoder.py       # 多尺度 Transformer 条件编码器
├── jump.py          # 自适应跳跃模块（针对非复合泊松的跳跃噪声）            
├── load_data.py     # 数据处理
├── loss.py          # 稳健异方差损失函数
├── model.py         # 整体扩散时间序列模型
├── main.py          # 主程序
├── trainer.py       # 训练、验证与采样的封装，高层 API
├── LICENSE          # MIT许可证
└── README.md        # 项目说明文档
```

## 安装

### 1. 通过 pip 安装

### 2. 通过源码安装

1. 克隆仓库

   ```bash
   git clone https://github.com/AWeizihao/DiffusionForTimeSeries.git
   cd DiffusionForTimeSeries
   ```
2. 本地安装

   使用以下命令进行可编辑方式安装：

   ```bash
   pip install -e .
   ```

## 使用方法

main.py 是一个使用示例。

您需要自行完成数据导入，并存入对象 `data` 中。推荐  `data` 以 Pandas DataFrame 格式储存。

要求是这样格式的 Pandas DataFrame：

| 索引列（可选） | 第 0 列 | 第 1 列 | ... |
| -------------- | ------- | ------- | --- |
| 2024-01-01     |  10.0       | 0.5        |     |
| 2024-01-02     |     10.2    |    0.6     |     |
| ...            |         |         |     |

注意，索引列必须曾显式指定 index 为某一列，否则不会自动存在索引列。保险起见无需加入索引列。

使用 Python 列表或 Numpy 数组是不推荐的。如一定要使用，需要满足以下格式（注意不要加入日期）

```python
# N 维列表
[
    [10.0, 0.5, ...], # 2025-01-01 
    [10.2, 0.6, ...], # 2025-01-02
    ...
]
# 或者 N 维 NumPy 数组
np.array([
    [ 10.0, 0.5, ...], # 2025-01-01
    [ 10.2, 0.6, ...], # 2025-01-02
    ...
])
```


参数表 `config` 目前保持默认参数，修改时**注意** `"target_dim"` 和 `"cond_input_dim"` 必须与输入数据表 `data` 需要学习的列数一致（不含索引列），否则有报错和截断的风险。

```python
# 参数配置字典
config = {
    "cond_len": 60, # 条件数据长度（模型记忆长度）
    "seq_len": 10,  # 需要模型预测的序列长度
    "target_dim": 9, # 需要预测的维度，需要与 target_fields 数量一致，或与 data 列数一致（不含索引列）
    "cond_input_dim": 9, # 输入数据的维度，需要与 target_fields 数量一致，或与 data 列数一致（不含索引列）
    "cond_dim": 32,  # 条件数据嵌入维度
    "time_embed_dim": 32,  # 时间嵌入维度
    "hidden_dim": 64,  # 隐藏层神经元个数
    "T_diffusion": 100, # 扩散步骤数
    "lr": 0.001,       # 学习率
    "weight_decay": 0.0001, # # 权重衰减
    "n_epochs": 8000, # 训练轮数
    "volatility": 0.5,  # 波动率
    "use_dynamic_beta": True,  # True 则使用余弦 Beta 计划，False 则为线性计划
    "s_base": 0.008, # 基础 S 值
    "alpha": 1.0, # alpha 参数
    "use_compound_poisson": True, # 是否使用复合泊松过程
    "jump_intensity": 0.05, # 跳跃强度，置为 0 则关闭跳跃扩散
    "jump_scale": 1.0, # 跳跃尺度
    "ddim_steps": 50, # DDIM 反扩散步骤数
    "eta": 0.1
}

from trainer import train_model

# 自行载入数据 data ，然后运行本程序即可。

# data 推荐格式为 Pandas DataFrame。如果使用 Pandas DataFrame 格式数据，请将时间列（如果有）设置为索引或直接删去。
# data 如使用字典、numpy数组，会尝试转化为 Pandas DataFrame。

target_fields=["log_return", "volume", "turnover", "high", "low", "EMA_20", "EMA_50", "RSI_14", "MACD"]

# 注意 target_fields 列表必须是 Pandas DataFrame 的列名（不包括索引），如果不匹配，则只会使用 target_fields 字段与 Pandas DataFrame 的列名重合的部分列。
# 为避免问题，可以直接不传入 target_field 参数（target_field 是 train_model 的可选参数）。

model = train_model(config,data)
```

## 数学结构

本模型以 **扩散模型（Diffusion Model）** 为核心，结合了 **DDIM** 采样、 **动态噪声调度（Dynamic Beta Schedule）** 、 **复合泊松/自适应跳跃噪声** 以及 **稳健异方差损失（Robust Heteroscedastic Loss）** 等多个要点。

### 1. 扩散过程的整体框架

扩散模型通过**前向过程**和**逆向过程**实现从真实数据到噪声、再从噪声回到数据的双向映射关系。设训练数据为 $\{\mathbf{x_0}\}$，我们令：

1. **前向过程：**

    将真实数据 $\{\mathbf{x_0}\}$ 逐步加噪得到 $\{\mathbf{x_1}, \mathbf{x_2}, \dots, \mathbf{x_T}\}$：

    $$\mathbf{x_t} = \sqrt{\bar{\alpha}_t}\, \mathbf{x_0} + \sqrt{1 - \bar{\alpha}_t}\,\epsilon,
    \quad \epsilon \sim \mathcal{N}(0, I)$$

    其中

    $$\alpha_t = 1 - \beta_t,
    \quad
    \bar{\alpha}_t = \prod_{i=0}^{t}\alpha_i$$

    $ \beta_t $ 为在每一步加噪时所使用的噪声权重计划。$ \beta_t $ 调度计划可以使用线性调度或动态调度。

2. **逆向过程：**
    希望学习一个去噪网络（或去噪分布）能够以 “去掉” 前向过程添加的噪声方式，反向得到对 $x_0$ 的推断，从而实现生成或预测。在最朴素的扩散模型（如DDPM）中，逆向过程可以通过数值迭代逼近

    $$p(x_{t-1} \mid x_t)$$

    在本程序中，通过**DDIM**（Denoising Diffusion Implicit Models）来进行更高效的逆向采样，减少采样步数并可控地注入噪声。


### 2. 动态余弦调度及其推导

在原始DDPM（Denoising Diffusion Probabilistic Models）框架中，$ \beta_t $ 通常可以是线性或指数递增。然而在实践中，使用**余弦调度**（cosine schedule）往往能获得更稳定的训练与更好的生成质量。其核心思路是：用余弦函数从 1 到 0 的变化映射来控制 $\bar{\alpha}_t$ 的变化速度。

1. **标准余弦调度：**
    令

    $$\bar{\alpha}_t = \cos^2\Bigl(\tfrac{\pi}{2}\cdot u_t\Bigr)$$

    其中 $u_t$ 通常随 $t/T$ 做线性变化，并且会结合一个平滑参数 $s$。

    $$
    u_t = \frac{\frac{t}{T} + s}{1 + s}.
    $$

    通过泰勒展开可证明，将 $ 𝑡/𝑇 $ 作为自变量时，余弦函数具有良好的平滑性。当 $ 𝑡→0$ $t→T$ 时：

    $$
    \bar{\alpha}_t \approx \cos^2\!\left[\frac{\pi}{2}\frac{s}{1+s}\right] \quad \text{或} \quad \bar{\alpha}_t \approx \cos^2\!\left[\frac{\pi}{2}\frac{1+s}{1+s}\right]
    $$

    从而有助于确保梯度在各步之间能够平稳传递。这为后续的反向传播提供理论保证。

    所以：

    $$
    \bar{\alpha}_t
    = \cos^2\Bigl(\frac{\pi}{2} \cdot \frac{\frac{t}{T}+s}{1+s}\Bigr).
    $$

    从而得到：

    $$
    \beta_t
    = 1 - \frac{\bar{\alpha}_{t+1}}{\bar{\alpha}_t}.
    $$

1. **动态余弦调度：**
   
    为适应不同数据集的波动特征，可引入序列的标准差来动态地修正 $s$。令

    $$
    s = s_{\text{base}} \times \bigl(1 + \alpha \cdot \text{volatility}\bigr).
    $$

    设计确保了当数据波动较大时，调度曲线能够自适应地平移。这不仅保证了前向过程噪声注入的灵活性，同时也在数值上起到平滑梯度的作用。这样当数据波动较大时，$\beta_t$ 的变化曲线也更快或更慢地覆盖初始与中间阶段，从而对大波动序列更加灵活。

    利用该动态调度后，前向过程的 “噪声注入量” $\beta_t$ 会自适应地进行放缩，有助于在高波动场景下保持更稳定或更适应的收敛性能。


### 3. 复合泊松过程与自适应跳跃噪声

现实金融数据常常存在“**跳跃**”或“黑天鹅事件”，即短时间内出现大幅度波动。为捕捉此类跳跃现象，可以在标准**高斯扩散**的基础上增加额外的“**跳跃噪声**”。

1. **复合泊松跳跃：**

    - 令泊松强度（下文记为 $\lambda$）控制单位时间对跳跃次数的期望。对每个样本在一步加噪时：

    $$
    n_{\mathrm{jumps}} \sim \mathrm{Poisson}(\lambda).
    $$

    - 如若 $n_{\mathrm{jumps}} > 0$，我们将其视为一次“聚合跳跃”，拥有

    $$
    \sqrt{n_{\mathrm{jumps}}}\cdot \mathrm{jump\_scale} \cdot \epsilon_{\mathrm{jump}},
    \quad
    \epsilon_{\mathrm{jump}}\sim \mathcal{N}(0,I).
    $$

    这样在同一时间步内若出现多次跳跃，等价于一次方差加大了的合并跳跃。

2. **自适应跳跃模块：**

    - 当不使用复合泊松，而是让神经网络自适应地学习“跳跃发生概率” $\mathrm{jump\_prob}$ 以及“跳跃幅度” $\mathrm{jump\_scale}$；

    - 在实践中可通过一层或多层全连接网络对条件编码器的输出 $h$ 做映射：

    $$
    [p,\,s] = f_\theta(h).
    $$

    - $p = \sigma(\mathrm{logits\_p}) \in [0,1]$：跳跃发生概率；
    - $s = \mathrm{Softplus}(\mathrm{logits\_s}) > 0$：跳跃幅度。
    - 每个样本每步加噪时，通过一次伯努利采样来决定是否加上该跳跃噪声。

    上述跳跃机制仅在序列某一通道（例如第 0 通道）上叠加保证使整体噪声分布在大部分情况下保持相对平稳，而在关键时刻出现较大增量，更符合某些金融时序的波动特征。



### 4. DDIM 逆向采样原理

DDIM（Denoising Diffusion Implicit Models）提出了一种更高效的采样方案，其核心是引入一个参数 $\eta$ 控制噪声注入量，既可使采样变为**确定性（ $\eta=0$ )**，也可保留一定随机性（$\eta>0$)。

1. **记号：**

   - $x_t$ 为扩散后的数据；
   - $\bar{\alpha}_t = \prod_{i=0}^{t}\alpha_i$；
   - 去噪网络预测的 $x_0^{\mathrm{pred}}$ 和噪声 $\epsilon_{\mathrm{pred}}$。其中

   $$
   \epsilon_{\mathrm{pred}} = \frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0^{\mathrm{pred}}}{\sqrt{1-\bar{\alpha}_t}}.
   $$

2. **DDIM更新公式：**

$$
x_{t-1}= \sqrt{\bar{\alpha}_{t-1}} \; x_0^{\mathrm{pred}}+ \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\;\epsilon_{\mathrm{pred}}+ \sigma_t \, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

其中

$$
\sigma_t = \eta \sqrt{
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
\left(
1- \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
\right)
}
$$

- 当 $\eta=0$ 时，$\sigma_t=0$，则每一步都是确定性的映射，实现**快速推断**；
- 当 $\eta>0$ 时，在逆向过程中仍会在各步注入少量额外噪声，使得采样更具多样性。



### 5. 稳健异方差损失函数

为了对**预测值本身**与其**波动风险**（不确定性）进行联合建模，网络输出不再只是预测的目标序列 $\hat{x}_0$，还包含一个额外的 $\log\sigma$（对角线近似，或与特征维度一致）。那么：

1. **平稳异方差思路：**

    $$
    p(x_0 \mid \theta) = \mathcal{N}\bigl(\hat{x}_0,\; \sigma^2 I\bigr),
    $$

    则极大似然在负对数似然意义下，相当于最小化

    $$
    \|\hat{x}_0 - x_0\|^2 / (2\sigma^2) + \log \sigma,
    $$

    这是一种常见的“可学习方差”思路。

2. **稳健（Huber）损失：**

    考虑到金融数据中可能有异常值或跳跃，均方误差可能过于敏感，故将 $\|\hat{x}_0 - x_0\|^2$ 改成 **Huber Loss**（Smooth L1）：

    $$
    \mathrm{Huber}(r) =
    \begin{cases}
    0.5r^2, & \text{if } |r| < \delta, \\
    \delta|r| - 0.5\delta^2, & \text{otherwise}.
    \end{cases}
    $$

    （在 PyTorch 中 `F.smooth_l1_loss` 已做了内部实现，缺省 $\delta=1$。）

    最终损失写成：

    $$
    \mathcal{L} =
    \frac{1}{2} \Bigl[\exp(-\log\sigma)\,\mathrm{Huber}(\hat{x}_0, x_0) + \log\sigma\Bigr].
    $$

    使得当数据误差较大时，“缩放因子” $\exp(-\log\sigma)$ 可以自动调节对异常值的影响，同时保留了在 $\mathrm{Huber}$ 范围内的平滑过渡。



## 代码解释

### 1. 动态噪声调度

```python
def get_dynamic_beta_schedule(T, volatility, s_base=0.008, alpha=1.0):
    """
    根据波动率调整余弦调度参数，生成 beta 序列。
    
    参数:
      T: 扩散步数
      volatility: 波动率指标（例如 log_return 的标准差）
      s_base: 基础 s 参数
      alpha: 控制波动率对 s 的影响因子
    
    返回:
      正则化后的 beta 张量
    """
    s = s_base * (1 + alpha * volatility)
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def get_beta_schedule(T, s=0.008):
    """
    原始静态 beta 调度函数。
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)
```

- 令 `x` 从 0 到 `T` 均匀分布；
- 先通过余弦函数得到 $\bar{\alpha}_t$（记为 `alphas_cumprod`），再转化为 $\beta_t$；
- 在此基础上通过 `torch.clamp` 控制 $\beta_t$ 的最小值与最大值，避免出现数值不稳定


### 2. 复合泊松与自适应跳跃的加噪实现（Diffusion类）

```python
class Diffusion(nn.Module):
    def __init__(self, T, volatility=0.0, use_dynamic_beta=True, s_base=0.008, alpha=1.0,
                 use_compound_poisson=False, jump_intensity=0.1, jump_scale=1.0):
        """
        扩散模块：
          - 可选动态 beta 调度与静态 beta 调度
          - 可选添加复合泊松跳跃噪声或使用自适应跳跃模块
          
          T: 扩散步数
          volatility: 波动率指标
          use_dynamic_beta: 是否使用动态 beta 调度函数（True/False）
          s_base, alpha: beta 调度参数控制因子
          use_compound_poisson: 是否使用复合泊松过程 (True/False)
          jump_intensity: 复合泊松跳跃参数 λ
          jump_scale: 跳跃尺度（标准差）
        """
        super().__init__()
        self.T = T
        if use_dynamic_beta:
            betas = get_dynamic_beta_schedule(T, volatility, s_base, alpha)
        else:
            betas = get_beta_schedule(T, s_base)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        
        # 复合泊松过程参数设置
        self.use_compound_poisson = use_compound_poisson
        self.jump_intensity = jump_intensity
        self.jump_scale = jump_scale
        
    def q_sample(self, x0, t, noise=None, jump_prob=None, jump_scale_adaptive=None):
        """
        根据当前扩散步 t 为输入 x0 添加噪声。
        当 use_compound_poisson 为 True 时，使用复合泊松跳跃噪声；
        否则根据 jump_prob 与 jump_scale_adaptive 添加自适应跳跃噪声。
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t]
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.view(1)
        x_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1) * x0 + torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1) * noise

        if self.use_compound_poisson:
            batch_size = x0.shape[0]
            n_jumps = torch.poisson(torch.full((batch_size, 1, 1), self.jump_intensity, device=x0.device))
            jump_noise = torch.sqrt(n_jumps) * self.jump_scale * torch.randn_like(x0[:, :, 0:1])
            x_t[:, :, 0:1] = x_t[:, :, 0:1] + jump_noise
        elif jump_prob is not None and jump_scale_adaptive is not None:
            batch_size = x0.shape[0]
            if isinstance(jump_prob, torch.Tensor):
                jump_prob = jump_prob.view(batch_size, 1, 1)
            else:
                jump_prob = torch.full((batch_size, 1, 1), jump_prob, device=x0.device)
            if isinstance(jump_scale_adaptive, torch.Tensor):
                jump_scale_adaptive = jump_scale_adaptive.view(batch_size, 1, 1)
            else:
                jump_scale_adaptive = torch.full((batch_size, 1, 1), jump_scale_adaptive, device=x0.device)
            jump_mask = (torch.rand(batch_size, 1, 1, device=x0.device) < jump_prob).float()
            jump_noise = torch.randn_like(x0[:, :, 0:1]) * jump_scale_adaptive
            x_t[:, :, 0:1] = x_t[:, :, 0:1] + jump_mask * jump_noise

        return x_t
```

1. 首先根据 $\bar{\alpha}_t$ 对 $x_0$ 做**标准扩散加噪**；
2. 如配置使用复合泊松，则按照泊松分布采样跳跃次数并叠加；
3. 否则，若给定了 `jump_prob` 和 `jump_scale_adaptive`，则进行自适应跳跃。在实现中只对 `x0[:, :, 0:1]` （即第0通道）加入跳跃噪声/通过 `torch.poisson`、`torch.rand` 等方式随机化跳跃次数与概率。利用 `jump_mask` 实现伯努利过程在是否加跳跃上的决策。


### 3. 多尺度条件编码器

```python
class MultiScaleConditionEncoder(nn.Module):
    def __init__(...):
    ...
    def forward(self, x):
        # 细尺度：Transformer
        fine = self.fc_in(x)
        fine = fine.transpose(0, 1)
        fine_transformed = self.transformer_encoder(fine)
        fine_transformed = fine_transformed.transpose(0, 1)
        fine_feat = fine_transformed.mean(dim=1)

        # 粗尺度：1D 卷积
        coarse_in = x.transpose(1, 2)
        coarse_feature = self.conv_coarse(coarse_in)
        coarse_feat = coarse_feature.mean(dim=2)

        # 拼接 & 输出
        fused = torch.cat([fine_feat, coarse_feat], dim=-1)
        cond = self.fc_merge(fused)
        return cond
```

- **原理**：
  - 细尺度分支借助 TransformerEncoder 捕捉长期（seq_len 维度）序列依赖，输出后对时序维度做 mean pooling；
  - 粗尺度分支用卷积（kernel_size=3, stride=2）进行下采样，能更直接地学习较粗的上下文特征，同样对输出做 mean pooling；
  - 最后将两者拼接后映射到 `cond_dim` 形成条件向量。

- **为什么要多尺度？**
  高频信息（短期波动）和低频趋势（相对平滑的长期趋势）都对金融市场预测或者时序预测非常关键。多尺度方法能在一个编码器中整合这两种信息。


### 4. 去噪网络及时间嵌入

去噪网络由 LSTM 构成，除输入扩散后的 $x_t$ 外，还需融合**时间嵌入**和**条件信息**。

```python
class TimeEmbedding(nn.Module):
    ...
    def forward(self, t):
        t_embed = get_sinusoidal_embedding(t, self.linear.in_features).to(t.device)
        return self.dropout(self.activation(self.linear(t_embed)))
```

- **Sinusoidal 嵌入**：与 Transformer 位置编码类似，频率从低到高的正弦与余弦基函数帮助网络捕捉随 `t` 的非线性周期性变化。

```python
class DirectDenoiserSequenceWithRisk(nn.Module):
...
    def forward(self, x_t, t_embed, h):
        t_proj = self.time_embed_proj(t_embed)
        t_proj_seq = t_proj.unsqueeze(1).repeat(1, seq_len, 1)
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1)
        x_concat = torch.cat([x_t, t_proj_seq, h_seq], dim=-1)

        lstm_out, _ = self.lstm(x_concat)
        ...
        x0_pred, log_sigma = torch.chunk(out, 2, dim=-1)
        return x0_pred, log_sigma
```

LSTM 输入是长度为 seq_len 的序列，每个时间步的特征拼接了：
1. 当前状态 $x_t$（含特征维度 feature_dim）；
2. 时间嵌入 $t_{\mathrm{embed}}$；
3. 条件向量 $h$（来自前文的多尺度编码器）。

输出维度拆分为两部分：
- $x_0^{\mathrm{pred}}$：预测的无噪声状态；
- $\log \sigma$：对数标准差，用于稳健异方差损失以及风险度量。


### 5. 采样流程（DDIM）

```python
def sample_ddim(self, c, ddim_steps=50, eta=0.0):
    x = torch.randn(batch_size, self.seq_len, self.feature_dim, device=device)
    ddim_timesteps = np.linspace(0, self.T_diffusion - 1, ddim_steps, dtype=int)[::-1]
    h = self.cond_encoder(c)

    for i in range(len(ddim_timesteps) - 1):
    t_val = ddim_timesteps[i]
    t_next = ddim_timesteps[i+1]

    t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
    t_embed = self.time_embed(t_tensor)

    x0_pred, log_sigma = self.denoiser(x, t_embed, h)
    alpha_bar_t = self.diffusion.alpha_bars[t_val]
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    ...
    # 计算 eps_pred
    eps_pred = (x - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
    ...
    # 计算 sigma_t, sqrt_coef
    x = sqrt_alpha_bar_t_next * x0_pred + sqrt_coef * eps_pred + sigma_t * noise

    risk_sigma = torch.exp(0.5 * last_log_sigma)
    return x, risk_sigma
```


- 初始 `x` 为标准正态分布的噪声；
- 通过 `linspace(...)[::-1]` 生成从 $T$ 到 0（近似）的逆向时间序列；
- 每一步中：
  1. 计算 `t_embed`（时间嵌入）；
  2. 调用去噪网络得出 $x_0^{\mathrm{pred}}, \log\sigma$；
  3. 根据 DDIM 公式计算 `eps_pred` 并更新下一个时刻的 `x`；
- 最终返回去噪后的序列 $x$ 以及风险标准差 `risk_sigma = exp(0.5 * log_sigma)`。


### 6. 训练流程与稳健异方差损失

```python
def robust_heteroscedastic_loss(x0_pred, log_sigma, x0_true):
    huber_loss = F.smooth_l1_loss(x0_pred, x0_true, reduction='none')
    loss = 0.5 * (torch.exp(-log_sigma) * huber_loss + log_sigma)
    return loss.mean()
```

训练时，对于每个批次样本，会随机采样一个时间步 `t`，执行一次正向扩散 `x_t` 并通过模型得到 $\hat{x}_0, \log\sigma$，然后计算上述损失。

该损失包括：
1. $\exp(-\log\sigma)\cdot \mathrm{HuberLoss}(\hat{x}_0, x_0)$：对误差进行对应的缩放；
2. $\log\sigma$：对学习到的方差做正则。

在金融等可能存在脉冲噪声或极端值的场景下，HuberLoss 能平衡均方误差和绝对误差的优点，且对异常样本更具鲁棒性。

