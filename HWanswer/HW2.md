@import "../style.less"

# 第二次作业解答

## 1. 证明题

> (1) 
> 
> Given conditions:
> 
>> (A1) The relationship between response (y) and covariates (X) is linear;
>> 
>> (A2) $\mathbf{X}$ is a non-stochastic matrix and rank($\mathbf{X}$) = $p$;
>> 
>> (A3) \(E(\varepsilon) = 0\). This implies \(E(y) = \mathbf{X} \beta\);
>> 
>> (A4) \(\mathrm{cov}(\varepsilon) = E(\varepsilon \varepsilon^\mathrm{T}) = \sigma^2 \boldsymbol{I}_N\); (Homoscedasticity);
>> 
>> (A5) \(\varepsilon\) follows multivariate normal distribution \(N(0, \sigma^2 \boldsymbol{I}_N)\) (Normality).
> 
> Prove the following results:
> 
>> (1.1) Prove that the OLS estimator \(\hat{\beta}\) is the same as the maximum likelihood estimator.
>> 
>> (1.2) Prove
>> \[
>> \hat{\beta} \sim N(\beta, \sigma^2 (X^\top X)^{-1})
>> \]
>> \[
>> (N - p)\hat{\sigma}^2 \sim \sigma^2 \chi^2_{N - p}
>> \]

### (1.1)
We know that $\hat{\beta}_{\mathrm{OLS}}=\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\mathbf{y}$, and according to $\varepsilon\sim N(0, \sigma^2 \boldsymbol{I}_N)$ we have
$$
\mathbf{y}\sim N\left( \mathbf{X}\beta ,\sigma ^2\boldsymbol{I} \right) .
$$

As a result the joint-PDF of $\mathbf{y}$ is 
$$
f\left( Y|\beta \right) =\frac{1}{\left( 2\pi \sigma ^2 \right) ^{N/2}}\exp \left( -\frac{1}{2\sigma ^2}\left( \mathbf{X}\beta -\varepsilon \right) ^{\mathrm{T}}\left( \mathbf{X}\beta -\varepsilon \right) \right) .
$$

Let
$$
L\left( \beta \right) =-\log f\left( Y|\beta \right) =\frac{N}{2}\log 2\pi \sigma ^2+\frac{1}{2\sigma ^2}\left( \mathbf{X}\beta -\varepsilon \right) ^{\mathrm{T}}\left( \mathbf{X}\beta -\varepsilon \right) ,
$$
and what we what to prove if $\hat{\beta}_{\mathrm{OLS}}=\mathop {\mathrm{arg}\min} \limits_{\beta}L\left( \beta \right)$, so
$$
\frac{\mathrm{d}}{\mathrm{d}\beta}L\left( \beta \right) =-2\mathbf{y}^{\mathrm{T}}\mathbf{X}-2\mathbf{X}^{\mathrm{T}}\mathbf{X}\beta \quad \Longrightarrow \quad \left. \frac{\mathrm{d}}{\mathrm{d}\beta}L\left( \beta \right) \right|_{\beta =\hat{\beta}_{\mathrm{OLS}}}=0.
$$

### (1.2)
We express $\hat{\beta}$ with subsitituting $\mathbf{y}=\mathbf{X}\beta +\varepsilon$,
$$
\hat{\beta}=\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\left( \mathbf{X}\beta +\varepsilon \right) =\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\mathbf{X}\beta +\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\varepsilon =\beta +\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\varepsilon .
$$

So
$$
E\left( \hat{\beta} \right) =\beta +\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}E\left( \varepsilon \right) =\beta ,
$$
$$
D\left( \hat{\beta} \right) =\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}D\left( \varepsilon \right) \left( \left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}} \right) ^{\mathrm{T}}=\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\sigma ^2\boldsymbol{I}\mathbf{X}\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}=\sigma ^2\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}.
$$

Finally $\hat{\beta} \sim N(\beta, \sigma^2 (X^\top X)^{-1})$. Cus we use $p$ degree of freedom in estimate $
\hat{\beta}$ so if we want to estimate $\hat{\sigma}^2$ we should devide $N-p$ instead of $N$,
$$
\hat{\sigma}^2=\frac{1}{N-p}\left( \mathbf{y}-\mathbf{X}\hat{\beta} \right) ^{\mathrm{T}}\left( \mathbf{y}-\mathbf{X}\hat{\beta} \right) =\frac{1}{N-p}\mathbf{e}^{\mathrm{T}}\mathbf{e}.
$$

Risidual vector $\mathbf{e}$ has the following property.
$$
\mathbf{e}:= \mathbf{y}-\mathbf{X}\hat{\beta}=\mathbf{y}-\mathbf{X}\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}}\mathbf{y}=\left( \mathbf{I}_N-\mathbf{P} \right) \mathbf{y}=\left( \mathbf{I}_N-\mathbf{P} \right) \varepsilon .
$$
Here $\mathbf{P}=\mathbf{X}\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^T$ is a projection matrix, as as $\mathbf{I}_N-\mathbf{P}$. And
$$
\begin{align*}
\left( \mathbf{I}_N-\mathbf{P} \right) \mathbf{y}&=\left( \mathbf{I}_N-\mathbf{P} \right) \left( \mathbf{X}\beta +\varepsilon \right) =\left( \mathbf{I}_N-\mathbf{P} \right) \mathbf{X}\beta +\left( \mathbf{I}_N-\mathbf{P} \right) \varepsilon 
\\
&=X\beta -\mathbf{X}\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^T\mathbf{X}\beta +\left( \mathbf{I}_N-\mathbf{P} \right) \varepsilon =\left( \mathbf{I}_N-\mathbf{P} \right) \varepsilon .
\end{align*}
$$
So 
$$
\hat{\sigma}^2=\frac{1}{N-p}\varepsilon ^T\left( \mathbf{I}_N-\mathbf{P} \right) \varepsilon.
$$

With $\mathrm{rank}\left( \mathbf{X} \right) =p$ we have $\mathrm{rank}\left( \mathbf{I}_N-\mathbf{P} \right) =N-p$. so
$$
\hat{\sigma}^2\sim \frac{\sigma ^2}{N-p}\chi _{N-p}^{2}.
$$

> (2) 
> Suppose \( y \) follows the log-linear regression relationship with non-stochastic \( x \in \mathbb{R}^p \), i.e.,
> $$\log(y) = x^\top \beta + \epsilon,$$
> where \( \epsilon \) follows normal distribution \( N(0, \sigma^2) \). Please calculate \( E(y) \).

Transform the expression for $y$,
$$
y=\exp \left( x^{\mathrm{T}}\beta +\epsilon \right) .
$$
So
$$
E\left( y \right) =E\left( \exp \left( x^{\mathrm{T}}\beta +\epsilon \right) \right) =\exp \left( x^{\mathrm{T}}\beta \right) E\left( \exp \epsilon \right) .
$$

To calculate $E\left( \exp \epsilon \right)$, 
$$
\begin{align*}
E\left( \exp \epsilon \right) &=\int_{\mathbb{R}}{\exp \epsilon \frac{1}{\sqrt{2\pi \sigma ^2}}\exp \left( -\epsilon ^2/2\sigma ^2 \right)}\frac{\mathrm{d}\varepsilon}{\mathrm{d}\exp \epsilon}\mathrm{d}\exp \epsilon 
\\
&=\exp \left( \sigma ^2/2 \right) \int_{\mathbb{R}}{\frac{1}{\sqrt{2\pi \sigma ^2}}\exp \left( -\left( \epsilon -\sigma ^2 \right) ^2/2\sigma ^2 \right)}\mathrm{d}\varepsilon =\exp \left( \sigma ^2/2 \right) 
\end{align*}
$$
As a result $E\left( y \right) =E\left( \exp \left( x^{\mathrm{T}}\beta +\epsilon \right) \right) =\exp \left( x^{\mathrm{T}}\beta +\frac{\sigma ^2}{2} \right) $.


> (3)
> Let \( y_i \) be the dependent variable, \( \boldsymbol{x}_i \) be the vector of independent variables including an intercept term, and \( \hat{\beta} \) be the vector of regression coefficients estimated by OLS. Define \( \hat{y}_i=\boldsymbol{x}_{i}^{\mathrm{T}}\hat{\beta} \). Define the total sum of squares (TSS), explained sum of squares (ESS), and residual sum of squares (RSS) as follows
> $$TSS = \sum_i (y_i - \bar{y})^2, \quad ESS = \sum_i (\hat{y}_i - \bar{y})^2, \quad RSS = \sum_i (y_i - \hat{y}_i)^2.$$
> Please prove: \( TSS = ESS + RSS \).

Insert $\hat{y}_i$ between the left hand side,
$$
\mathrm{TSS}=\sum_i{\left( y_i-\bar{y} \right) ^2}=\sum_i{\left[ \left( y_i-\hat{y}_i \right) +\left( \hat{y}_i-\bar{y} \right) \right] ^2=\mathrm{RSS}+\mathrm{ESS}+2\sum_i{\left( y_i-\hat{y}_i \right)}}\left( \hat{y}_i-\bar{y} \right),
$$
so we need to prove
$$
\sum_i{\left( y_i-\hat{y}_i \right) \left( \hat{y}_i-\bar{y} \right)}=0.
$$

First,
$$
\sum_i{\left( y_i-\hat{y}_i \right)}\bar{y}=\bar{y}\sum_i{e_i=0}.
$$

And we only need to prove
$$
\sum_i{\left( y_i-\hat{y}_i \right) \hat{y}_i}=\mathbf{e}^{\mathrm{T}}\hat{y}=0.
$$
And 
$$
\mathbf{e}^{\mathrm{T}}\hat{y}=\mathbf{e}^{\mathrm{T}}\mathbf{X}\hat{\beta}=\left( \mathbf{I}-\mathbf{X}\left( \mathbf{X}^{\mathrm{T}}\mathbf{X} \right) ^{-1}\mathbf{X}^{\mathrm{T}} \right) ^{\mathrm{T}}\mathbf{X}\hat{\beta}=0\hat{\beta}=0.
$$

Q.E.D.






## 2. 岭回归分析
> 在实际问题中，我们常常会遇到样本容量相对较小，而特征很多的场景. 在这种情况下如果直接求解线性回归模型，较小的样本无法支持带有很多模型参数、且有多个模型能够“完全”拟合训练集中的所有数据点. 此外，模型很容易出现过拟合. 为缓解这些问题，常在线性回归的损失函数中引入正则化项 \( p(\beta) \)，通常形式如下：
> $$\hat{\beta} = \arg \min_{\beta} \left\{ \sum_{i=1}^{N} \left( y_i - \sum_{j} x_{ij} \beta_j \right)^2 + \lambda p(\beta) \right\}$$
> 其中，\( \lambda > 0 \) 为正则化参数. 正则化表示对模型的一种偏好，例如 \( p(\beta) \) 一般取模型的复杂度进行约束，它在保持良好预测性能的同时，倾向于选择较为简单的模型，从而帮助防止过拟合并提高模型的泛化能力. 考虑岭回归（ridge regression）问题，即设置公式（4）中正则项 \( p(\beta) = \sum_j \beta_j^2 \). 本题将针对该回归问题的显式解以及正则化的影响进行探讨.
>
> (1) 请证明岭回归的最优解 \( \hat{\beta}^{\text{ridge}} \) 的显式解表达式具有以下两种等价形式：
> $$\hat{\beta}^{\text{ridge}} = \left( X^\top X + \lambda I_p \right)^{-1} X^\top y = X^\top \left( X X^\top + \lambda I_N \right)^{-1} y.$$
> 请分析以上两种最优解分别在何种情况下计算速度更快？
>
> (2) 分析岭回归的最优解 \( \hat{\beta}^{\text{ridge}} \) 和最小二乘估计 \( \hat{\beta}^{\text{LS}} \) 的区别？
>
> (3) 针对实际中描述的上述残差现象，完成以下任务:
> 
> (3.1) 完成数据读入与汇总统计，绘制训练集数据中月租金（rent）的直方图，观察月租金的大致分布，并进行简要解读；绘制训练集数据中月租金（rent）-城区（region）分组箱线图，分析不同城区的房价差异，并给出简要解读.
> 
> (3.2) 利用训练集建立以月租金（rent）为因变量，去除人为变量的线性回归模型，编程实现最小二乘估计（不调用回归分析的包），写出拟合得到的模型并计算训练集上的均方误差（Mean Square Error, MSE）.
>
> (3.3) 编程实现岭回归的回归计（不调用回归分析的包），在训练集上使用十折交叉验证，画出验证集上平均均方误差（Mean Square Error, MSE）与 \( \lambda \) 的折线图，选出合适的 \( \lambda \).
>
> (3.4) 用选出的 \( \lambda \) 在训练集拟合最终模型，写出拟合得到的模型并计算测试集上的均方误差.



