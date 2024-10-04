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





## 2. 岭回归分析

