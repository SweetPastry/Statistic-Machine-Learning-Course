# 第一次作业解答
<p align="center">姓名: 林海轩 &nbsp 学号: 2330711026 &nbsp 院系: 物理学系 &nbsp 专业: 物理学</p align="center">

## Q1
> 通过经验风险最小化推导极大似然估计. 证明模型是条件概率分布, 当损失函数是对数损失函数时, 经验风险最小化等价于极大似然估计.

即证明, 模型参数 $\theta$ 使得

$$
\underset{\theta}{\arg}\min\frac{1}{n}\sum_{i=1}^{n}L(y_i,f(x_i,\theta))=\underset{\theta}{\arg}\max p\left(\left.\prod_{i=1}^{n}Y_i\right|\prod_{i=1}^{n}X_i,\theta\right).
$$

其中, $x_i$ 与 $y_i$ 分别表示第 $i$ 次输入数据与期望的输出数据, $X_i$ 与 $Y_i$ 分别表示事件第 $i$ 次输入 $x_i$ 与输出 $y_i$. 不妨取损失函数为负对数, 则

$$
\begin{align*}
\mathrm{LHS}&=\underset{\theta}{\arg}\min\frac{1}{n}\sum_{i=1}^{n}L(y_i,f(x_i,\theta))
\\
&=\underset{\theta}{\arg}\max\frac{1}{n}\sum_{i=1}^{n}\log p(y_i,f(x_i,\theta))
\\
&=\underset{\theta}{\arg}\max\frac{1}{n}\log\prod_{i=1}^{n}p(y_i,f(x_i,\theta)).
\end{align*}
$$

此处再接受一个假设: **$\boldsymbol{\{x_n\}}$ 是独立的**.

$$
\begin{align*}
\mathrm{LHS}&=\underset{\theta}{\arg}\max\frac{1}{n}\log p\left(\left.\prod_{i=1}^{n}Y_i\right|\prod_{i=1}^{n}X_i,\theta\right).
\end{align*}
$$

注意到此时表达式显然与待证表达式的右侧相等. 得证.

## Q2
> 证明 Hoeffding 引理:
> 随机变量 $X$ 满足 $E(X)=0$ 且 $P(X\in\left[a,b\right])=1$, 则
> $$E\left(\mathrm{e}^{sX}\right)\leqslant\mathrm{e}^{\frac{s^2(b-a)^2}{8}}.$$

此处利用 $\mathrm{e}^{sx}$ 是下凸函数的性质, 有

$$
\mathrm{e}^{s\left( \theta a+\left( 1-\theta \right) b \right)}\leqslant \theta \mathrm{e}^{sa}+\left( 1-\theta \right) \mathrm{e}^{s\left( 1-\theta \right) b}.\qquad (0\leqslant\theta\leqslant 1)
$$

取 $\theta =\frac{b-X}{b-a}$, 得到

$$
\mathrm{e}^{sX}\leqslant \frac{b-X}{b-a}\mathrm{e}^{sa}+\frac{X-a}{b-a}\mathrm{e}^{sb}.
$$

两边求期望,

$$
E\left( \mathrm{e}^{sX} \right) \leqslant \frac{b-E\left( X \right)}{b-a}\mathrm{e}^{sa}+\frac{E\left( X \right) -a}{b-a}\mathrm{e}^{sb}=\frac{b}{b-a}\mathrm{e}^{sa}-\frac{a}{b-a}\mathrm{e}^{sb}.
$$

这事实上是比 Hoeffding 引理更紧的不等式, 也就是说为完成此题, 只需要证明:

$$
\frac{b}{b-a}\mathrm{e}^{sa}-\frac{a}{b-a}\mathrm{e}^{sb}\leqslant \mathrm{e}^{\frac{s^2\left( b-a \right) ^2}{8}}.
$$

记

$$
F\left( s \right) =-\frac{b}{b-a}\mathrm{e}^{sa}+\frac{a}{b-a}\mathrm{e}^{sb}+\mathrm{e}^{\frac{s^2\left( b-a \right) ^2}{8}}.
$$

那么

$$
F^{\prime}\left( s \right) =\frac{ab}{b-a}\left( \mathrm{e}^{sb}-\mathrm{e}^{sa} \right) +\frac{s\left( b-a \right) ^2}{4}\mathrm{e}^{\frac{s^2\left( b-a \right) ^2}{8}}.
$$

当 $s<0$ 时 $F^{\prime}\left( s \right)<0$, 当 $s>0$ 时 $F^{\prime}\left( s \right)>0$, 于是

$$
F\left( s \right) \geqslant F\left( 0 \right) =0.
$$

这样就证明了此题.

## Q3
> 已知针对模型 $f$, 使用训练数据集得到的估计记为 $\hat{f}$, 现有独立于训练数据集的 $(x_0, y_0)$, 其中 $x_0$ 为非随机的给定值, $y_0 = f(x_0) + \varepsilon$, 其中 $\varepsilon$ 为随机误差项, 证明：
> $${E}\left( y_0 - \hat{f}(x_0) \right)^2 = \text{Var}\left( \hat{f}(x_0) \right) + \left[ \text{Bias} \left( \hat{f}(x_0) \right) \right]^2 + \text{Var}(\varepsilon).$$

这个等式学名是**偏差-方差分解 (Bias-Variance Decomposition)**, 在证明这个等式之前先证明一个小的引理:

>> $$E\left[ \left( f\left( x_0 \right) -\hat{f}\left( x_0 \right) \right) ^2 \right] =\mathrm{Var}\left( \hat{f}\left( x_0 \right) \right) +\mathrm{Bias}^2\left( \hat{f}\left( x_0 \right) \right) .$$

将等号左边展开写:

$$
\begin{align*}
&E\left( f^2\left( x_0 \right) +\hat{f}^2\left( x_0 \right) -2f\left( x_0 \right) \hat{f}\left( x_0 \right) \right) =E\left( f^2\left( x_0 \right) \right) +E\left( \hat{f}^2\left( x_0 \right) \right) -2E\left( f\left( x_0 \right) \hat{f}\left( x_0 \right) \right) 
\\
&=f^2\left( x_0 \right) +E\left( \hat{f}^2\left( x_0 \right) \right) -2f\left( x_0 \right) E\left( \hat{f}\left( x_0 \right) \right) 
\\
&=\left( E\left( \hat{f}^2\left( x_0 \right) \right) -E^2\left( \hat{f}\left( x_0 \right) \right) \right) +\left( f^2\left( x_0 \right) -2f\left( x_0 \right) E\left( \hat{f}\left( x_0 \right) \right) +E^2\left( \hat{f}\left( x_0 \right) \right) \right) 
\\
&=\left( E\left( \hat{f}^2\left( x_0 \right) \right) -E^2\left( \hat{f}\left( x_0 \right) \right) \right) +\left( f\left( x_0 \right) -E\left( \hat{f}\left( x_0 \right) \right) \right) ^2
\\
&=\mathrm{Var}\left( \hat{f}\left( x_0 \right) \right) +\mathrm{Bias}^2\left( \hat{f}\left( x_0 \right) \right) 
\end{align*}
$$

接下来是原问题的证明. 

$$
\begin{align*}
 E\left( y_0-\hat{f}\left( x_0 \right) \right) ^2&=E\left( f\left( x_0 \right) -\hat{f}\left( x_0 \right) +\varepsilon \right) ^2
\\
&=E\left( \left( f\left( x_0 \right) -\hat{f}\left( x_0 \right) \right) ^2+2\varepsilon \left( f\left( x_0 \right) -\hat{f}\left( x_0 \right) \right) +\varepsilon ^2 \right) 
\\
&=E\left( f\left( x_0 \right) -\hat{f}\left( x_0 \right) \right) ^2+2E\left( \varepsilon \right) E\left( f\left( x_0 \right) -\hat{f}\left( x_0 \right) \right) +E\left( \varepsilon ^2 \right) 
\\
&=\mathrm{Var}\left( \hat{f}\left( x_0 \right) \right) +\mathrm{Bias}^2\left( \hat{f}\left( x_0 \right) \right) +E\left( \varepsilon ^2 \right)    
\end{align*}
$$

其中用到了随机误差项 $\varepsilon$ 的性质 $E(\varepsilon)=0$. 

## Q4
> 令 $\mathbf{y}=\mathbf{\Psi }\left( \mathbf{x} \right) $, 其中 $\mathbf{y}$ 是 $m\times 1$ 阶向量, $\mathbf{x}$ 是 $n\times 1$ 阶向量, 记
> $$\frac{\partial \mathbf{y}}{\partial \mathbf{x}^T}=\left( \begin{matrix}
	\frac{\partial y_1}{\partial x_1}&		\frac{\partial y_1}{\partial x_2}&		\cdots&		\frac{\partial y_1}{\partial x_n}\\
	\frac{\partial y_2}{\partial x_1}&		\frac{\partial y_2}{\partial x_2}&		\cdots&		\frac{\partial y_2}{\partial x_n}\\
	\vdots&		\vdots&		\,\,&		\vdots\\
	\frac{\partial y_m}{\partial x_1}&		\frac{\partial y_m}{\partial x_2}&		\cdots&		\frac{\partial y_m}{\partial x_n}\\
> \end{matrix} \right) .$$
> (a) 令 $\mathbf{y}=\mathbf{Ax}$, 其中 $\mathbf{A}$ 是 $m\times n$ 阶的矩阵, 且独立于 $\mathbf{x}$, 证明:
> $$\frac{\partial \mathbf{y}}{\partial \mathbf{x}^T}=\mathbf{A}.$$
> (b) 令标量 $\alpha=\mathbf{y}^T\mathbf{Ax}$, $\mathbf{A}$ 独立于 $\mathbf{x}$ 与 $\mathbf{y}$, 证明:
> $$
\frac{\partial \alpha}{\partial \mathbf{x}^T}=\mathbf{x}^T\mathbf{A}^T\left( \frac{\partial \mathbf{y}}{\partial \mathbf{x}^T} \right) +\mathbf{y}^T\mathbf{A}.
$$
> (c) 考虑一种特别的情况, 标量 $\alpha$ 由二次其次多项式给出:
> $$
\alpha =\mathbf{x}^T\mathbf{Ax}.
$$
> 其中 $\mathbf{A}$ 的形状是 $n\times n$, 且独立于 $\mathbf{x}$, 证明:
> $$
\frac{\partial \alpha}{\partial \mathbf{x}^T}=\mathbf{x}^T\left( \mathbf{A}+\mathbf{A}^T \right) .
$$
> (d) 令 $\alpha =\mathbf{y}^T\mathbf{Ax}$, 其中 $\mathbf{A}$ 的形状是 $m\times n$, $\mathbf{y}$ 和 $\mathbf{x}$ 都是 $\mathbf{z}$ 的函数, 其中 $\mathbf{z}$ 是 $q\times 1$ 阶向量, 且 $\mathbf{A}$ 独立于 $\mathbf{z}$. 证明:
> $$
\frac{\partial \alpha}{\partial \mathbf{z}^T}=\mathbf{x}^T\mathbf{A}^T\left( \frac{\partial \mathbf{y}}{\partial \mathbf{z}^T} \right) +\mathbf{y}^T\mathbf{A}\left( \frac{\partial \mathbf{x}}{\partial \mathbf{z}^T} \right) .
$$
> (e) 令 $\mathbf{A}$ 非奇异, 形状为 $m\times m$, 且其中的所有元素都是标量 $\alpha$ 的函数. 证明:
> $$
\frac{\partial \mathbf{A}^{-1}}{\partial \alpha}=-\mathbf{A}^{-1}\frac{\partial \mathbf{A}}{\partial \alpha}\mathbf{A}^{-1}.
$$

## Q5
> 求 $\hat{\mathbf{a}}$ 使得
> $$
\underset{\mathbf{a}}{\min}\left\| \mathbf{Xa}-\mathbf{y} \right\| _2.
$$
> 其中 $\mathbf{X}$ 是 $n\times p$ 阶的矩阵, $\mathbf{y}$ 是 $n\times 1$ 阶的向量, $\mathbf{a}$ 是 $p\times 1$ 阶的向量. $\mathbf{X}^T\mathbf{X}$ 是非奇异的.