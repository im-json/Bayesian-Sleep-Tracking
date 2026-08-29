# Bayesian Sleep Tracking
Autoregressive Poisson model using Metropolis-within-Gibbs MCMC

**Background**

$\texttt{Sleep}$ is defined as number of hours of sleep.

$\texttt{Tiredness}$ is defined as subjective tiredness on a scale from 1 to 10.

$\texttt{Fasting}$ is a factor variable, defined as if three quarters or more of a 24 hour day (18+ hours) were spent fasted.

# AR(1) model

Let $y$ be $\texttt{Tiredness}$ and $x_1$ be $\texttt{Sleep}$. For observations $t = 1,\dots,T$,

$$y_t \mid \eta_t \sim \text{Poisson}(\lambda_t), \quad \lambda_t = \exp(\eta_t)$$

$$\mu_t = \beta_0 + \beta_1x_{t1}$$

Let $\eta_t$ be the latent log-mean process.

$$\eta_t = \mu_t + \varphi(\eta_{t-1} - \mu_{t-1}) + \varepsilon_t$$

$$\varepsilon_t \sim \mathcal{N}\left(0,\sigma^2\right), \quad t = 2,\dots,T$$

Let $u_t = \varphi u_{t-1} + \varepsilon_t$.

$$u_1 \sim \mathcal{N}\left(0,\frac{\sigma^2}{1-\varphi^2}\right), \quad \eta_1 = \mu_1 + u_1$$

$$u_t \sim \mathcal{N}\left(\varphi u_{t-1},\sigma^2\right), \quad t = 2,\dots,T$$

**Likelihood**

The likelihood for $y_t$ is given by

$$L(y_t) = \prod_{t=1}^{T}\frac{\lambda_t^{y_t}e^{-\lambda_t}}{y_t!} = \frac{1}{\prod_{t=1}^{T}y_t!}\prod_{t=1}^{T}\exp\left\lbrace y_t\eta_t - e^{\eta_t}\right\rbrace = \frac{1}{\prod_{t=1}^{T}y_t!}\exp\left\lbrace\sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right)\right\rbrace$$

**Latent process**

The density for $u_1$ is given by

$$
p(u_1) = \frac{1}{\sqrt{2\pi\left(\sigma^2/\left(1-\varphi^2\right)\right)}}\exp\left\lbrace -\frac{u_1^2}{2\left(\sigma^2/\left(1-\varphi^2\right)\right)}\right\rbrace = \frac{\sqrt{1-\varphi^2}}{\sqrt{2\pi\sigma^2}}\exp\left\lbrace -\frac{u_1^2\left(1-\varphi^2\right)}{2\sigma^2}\right\rbrace
$$

The likelihood for $u_t$ is given by

$$L(u_t) = \prod_{t=2}^{T}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left\lbrace -\frac{(u_t - \varphi u_{t-1})^2}{2\sigma^2}\right\rbrace = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^{T}\prod_{t=2}^{T}\exp\left\lbrace -\frac{(u_t - \varphi u_{t-1})^2}{2\sigma^2}\right\rbrace$$

Hence we have

$$
\begin{aligned}
p(u_1)L(u_t) &= \frac{\sqrt{1-\varphi^2}}{\sqrt{2\pi\sigma^2}}\exp\left\lbrace -\frac{u_1^2\left(1-\varphi^2\right)}{2\sigma^2}\right\rbrace\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^{T}\prod_{t=2}^{T}\exp\left\lbrace -\frac{(u_t - \varphi u_{t-1})^2}{2\sigma^2}\right\rbrace \\
&= \frac{\sqrt{1-\varphi^2}}{\left(2\pi\sigma^2\right)^{(1+T)/2}}\exp\left\lbrace -\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}(u_t - \varphi u_{t-1})^2\right)\right\rbrace
\end{aligned}
$$

**Priors**

We choose the following weakly informative priors.

$$\beta_0 \sim \mathcal{N}\left(0,1^2\right), \qquad \beta_1 \sim \mathcal{N}\left(0,0.5^2\right), \qquad \varphi \sim \text{Uniform}(0,1), \qquad \sigma \sim \text{Half-Cauchy}(0,1)$$

The densities for $\beta_0$ and $\beta_1$ are

$$p(\beta_0) = \frac{1}{\sqrt{2\pi}}\exp\left\lbrace -\frac{\beta_0^2}{2}\right\rbrace, \qquad p(\beta_1) = \frac{1}{\sqrt{\pi/2}}\exp\left\lbrace -2\beta_1^2\right\rbrace$$

The density for $\varphi$ is given by

$$p(\varphi) = \frac{1}{b-a} = 1, \quad \varphi \in [0,1]$$

The density for $\sigma$ is given by

$$p(\sigma) = \frac{2}{\pi\gamma\left(1 + (\sigma/\gamma)^2\right)} = \frac{2}{\pi\left(1 + \sigma^2\right)}, \quad \sigma \ge 0$$

Hence we have

$$
\begin{aligned}
p(\beta_0)p(\beta_1)p(\varphi)p(\sigma) &= \frac{1}{\sqrt{2\pi}}\exp\left\lbrace -\frac{\beta_0^2}{2}\right\rbrace\frac{1}{\sqrt{\pi/2}}\exp\left\lbrace -2\beta_1^2\right\rbrace\frac{2}{\pi\left(1 + \sigma^2\right)} \\
&= \frac{2}{\pi^2\left(1 + \sigma^2\right)}\exp\left\lbrace -\frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace
\end{aligned}
$$

**Joint Posterior**

Hence the joint posterior is given by

$$
\begin{aligned}
p(\theta,\eta\mid y) &\propto \underbrace{\frac{1}{\prod_{t=1}^{T}y_t!}\exp\left\lbrace\sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right)\right\rbrace}\_{\text{Poisson likelihood}} \\
&\times \underbrace{\frac{\sqrt{1-\varphi^2}}{\left(2\pi\sigma^2\right)^{(1+T)/2}} \times \exp\left\lbrace-\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}(u_t - \varphi u_{t-1})^2\right)\right\rbrace}\_{\text{Latent process}} \\
&\times \underbrace{\frac{2}{\pi^2\left(1 + \sigma^2\right)}\exp\left\lbrace-\frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace}\_{\text{Priors}}
\end{aligned}
$$

We group constants into a single coefficient, and drop them while maintaining proportionality.

$$
\begin{aligned}
p(\theta,\eta\mid y) &\propto \left(\frac{2}{\pi^2(2\pi)^{(1+T)/2}\prod_{t=1}^{T}y_t!}\right)\exp\left\lbrace \sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right)\right\rbrace \times \frac{\sqrt{1-\varphi^2}}{\sigma^{1+T}} \\
&\times \exp\left\lbrace -\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}(u_t - \varphi u_{t-1})^2\right)\right\rbrace \times \frac{1}{\left(1 + \sigma^2\right)}\exp\left\lbrace -\frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace \\
p(\theta,\eta\mid y) &\propto \frac{\sqrt{1-\varphi^2}}{\left(1+\sigma^2\right)\sigma^{1+T}} \\
&\times \exp\left\lbrace \sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right) -\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}\left(u_t - \varphi u_{t-1}\right)^2\right) - \frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace
\end{aligned}
$$

**Log Posterior**

Hence the log posterior up to a proportionality is given by

$$
\begin{aligned}
\log\left(p(\theta,\eta\mid y)\right) &\propto \frac{1}{2}\log\left(1-\varphi^2\right) - \log\left(1+\sigma^2\right) - (1+T)\log(\sigma) + \sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right) \\
&- \frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}\left(u_t - \varphi u_{t-1}\right)^2\right) - \frac{\beta_0^2}{2} - 2\beta_1^2
\end{aligned}
$$

# Metropolis-within-Gibbs MCMC

**Pseudocode**

The following pseudocode is based off an example from RPubs.

1. Select an initial value $\theta\_0$.
2. For $i = 1,\dots,n$ iterations, repeat the following steps:

   (a) Set $\theta\_{i,0} = \theta\_{i-1}$

   (b) For $j = 1,\dots,4$, repeat the following steps:
     - Draw a candidate value $\theta^{(j)\*}$ from a proposal distribution $q\left(\theta^{(j)\*}\mid\theta\_{i,j-1}^{(j)}\right)$.
     - Form the candidate state $\theta\_{i,j}^\*$ by replacing the $j$th component of $\theta\_{i,j-1}$ with $\theta^{(j)\*}$, leaving all other components unchanged.
     - Compute the ratio
$$\alpha = \frac{g\left(\theta\_{i,j}^\*\right)/q\left(\theta^{(j)\*}\mid\theta\_{i,j-1}^{(j)}\right)}{g\left(\theta\_{i,j-1}\right)/q\left(\theta\_{i,j-1}^{(j)}\mid\theta^{(j)\*}\right)} = \frac{g\left(\theta\_{i,j}^\*\right)q\left(\theta\_{i,j-1}^{(j)}\mid\theta^{(j)\*}\right)}{g\left(\theta\_{i,j-1}\right)q\left(\theta^{(j)\*}\mid\theta\_{i,j-1}^{(j)}\right)}$$
     - If $\alpha \ge 1$, set $\theta\_{i,j} = \theta\_{i,j}^\*$. If $\alpha < 1$, then set $\theta\_{i,j} = \theta\_{i,j}^\*$ with probability $\alpha$, or $\theta\_{i,j} = \theta\_{i,j-1}$ with probability $1-\alpha$.

   (c) Set $\theta\_i = \theta\_{i,4}$

**Variables**

Let $\texttt{y}$ be $\texttt{Tiredness}$, $\texttt{x1}$ be $\texttt{Sleep}$ scaled between 0 and 1, and $\texttt{eta}$ be $\texttt{log(y)}$.

Let $\texttt{params}$ be a vector containing named parameters $\texttt{beta0, beta1, phi, sigma}$.

Let $\texttt{sd\\_cand}$ be a vector containing proposed standard deviations for each parameter.

```R
y <- Tiredness
x1 <- scale(Sleep)[,1]
eta_init <- log(y)
params <- c(beta0 = 1.0, beta1 = -0.3, phi = 0.5, sigma = 0.5,
            setNames(eta_init, paste0("eta", 1:length(eta_init))))
sd_cand <- c(beta0 = 0.1, beta1 = 0.1, phi = 0.05, sigma = 0.05,
             setNames(rep(0.3, length(y)), paste0("eta", 1:length(y))))
```

**Log Posterior Function**

We define an $\texttt{R}$ function $\texttt{log\\_post}$ which calculates the log posterior.

```R
log_post <- function(y, x1, params) {
  T <- length(y)
  
  beta0 <- params["beta0"]
  beta1 <- params["beta1"]
  phi <- params["phi"]
  sigma <- params["sigma"]
  eta <- params[paste0("eta", 1:T)]
  
  if (phi <= 0 || phi >= 1 || sigma <= 0) return (-Inf)
  
  mu <- beta0 + beta1*x1
  u <- eta - mu
  
  like <- sum(y*eta - exp(eta))
  latent <- -(1/(2*sigma^2))*((u[1]^2)*(1-phi^2) + sum((u[2:T] - phi*u[1:(T-1)])^2))
  priors <- (1/2)*log(1-phi^2) - log(1+sigma^2) - (1+T)*log(sigma) - (beta0^2/2) - 2*beta1^2
  
  return (like + latent + priors)
}
```

**Sampling Function**

Let $\texttt{n\\_iter}$ be the number of iterations. We define a function $\texttt{metro\\_within\\_gibbs}$ to run the random walk Metropolis-within-Gibbs algorithm.

```R
metro_within_gibbs <- function(y, x1, params, n_iter, sd_cand) {
  chain <- matrix(NA, nrow = n_iter, ncol = length(params),
                  dimnames = list(NULL, names(params)))
  accept <- setNames(numeric(length(params)), names(params))
  log_curr <- log_post(y, x1, params)
  
  for (i in 1:n_iter) {
    for (j in 1:length(params)) {
      cand <- params
      cand[j] <- params[j] + rnorm(1, mean = 0, sd = sd_cand[j])
      log_cand <- log_post(y, x1, cand)
      log_alpha = log_cand - log_curr
      
      if (log(runif(1)) < log_alpha) {
        params <- cand
        log_curr <- log_cand
        accept[j] <- accept[j] + 1
      }
    }
    chain[i,] <- params
  }
  
  list(chain = chain,
       prior_accept_rates = accept[c("beta0","beta1","phi","sigma")] / n_iter,
       eta_accept_rates = accept[grepl("^eta", names(accept))] / n_iter)
}
```

We run the sampling function for 100000 iterations.

```R
result <- metro_within_gibbs(y, x1, params, 100000, sd_cand)

result$prior_accept_rates
  beta0   beta1     phi   sigma 
0.32074 0.19566 0.89437 0.24970

result$eta_accept_rates
   eta1    eta2    eta3    eta4    eta5    eta6    eta7    eta8    eta9   eta10 
0.20180 0.18963 0.18811 0.19067 0.18893 0.19030 0.19004 0.18981 0.19246 0.20758
```

**Interpretation**

$\texttt{params}$ contains four parameters, and $\texttt{accept}$ contains their corresponding acceptance rates. For the one dimensional case, the optimal acceptance rate is approximately between 0.2 and 0.5. For parameters with acceptance rate $> 0.5$, we increase the corresponding values in $\texttt{sd\\_cand}$. For parameters with acceptance rate $< 0.2$, we decrease the corresponding values in $\texttt{sd\\_cand}$. After repeated adjustments, we get the following values for $\texttt{sd\\_cand}$.

```R
sd_cand <- c(beta0 = 0.05, beta1 = 0.0125, phi = 0.42, sigma = 0.0125,
             setNames(rep(0.075, length(y)), paste0("eta", 1:length(y))))

result <- metro_within_gibbs(y, x1, params, 100000, sd_cand)

result$prior_accept_rates
  beta0   beta1     phi   sigma 
0.39432 0.45894 0.45319 0.42155

result$eta_accept_rates
   eta1    eta2    eta3    eta4    eta5    eta6    eta7    eta8    eta9   eta10
0.38745 0.37209 0.37033 0.37488 0.36969 0.37096 0.36840 0.37157 0.37306 0.38811
```

**Burn-in**

# Inference
