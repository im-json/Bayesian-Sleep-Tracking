# Bayesian Sleep Tracking
Autoregressive Poisson model with examples

**Background**

$\texttt{Sleep}$ is defined as number of hours of sleep.

$\texttt{Tiredness}$ is defined as subjective tiredness on a scale from 1 to 10.

$\texttt{Fasting}$ is a factor variable, defined as if three quarters or more of a 24 hour day (18+ hours) were spent fasted.

**AR(1) model**

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

**Latent AR(1) process**

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
p(\beta_0)p(\beta_1)p(\varphi)p(\sigma) &= \frac{1}{\sqrt{2\pi}}\exp\left\lbrace -\frac{\beta_0^2}{2}\right\rbrace\frac{1}{\sqrt{\pi/2}}\exp\left\lbrace -2\beta_1^2\right\rbrace\frac{2}{\pi\left(1 + \sigma^2\right)} = \frac{2}{\pi^2\left(1 + \sigma^2\right)}\exp\left\lbrace -\frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace
\end{aligned}
$$

**Joint Posterior**

$$
\begin{aligned}
p(\theta,\eta\mid y) &\propto \frac{1}{\prod_{t=1}^{T}y_t!}\exp\left\lbrace \sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right)\right\rbrace \times \frac{\sqrt{1-\varphi^2}}{\left(2\pi\sigma^2\right)^{(1+T)/2}} \\
&\times \exp\left\lbrace -\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}(u_t - \varphi u_{t-1})^2\right)\right\rbrace \times \frac{2}{\pi^2\left(1 + \sigma^2\right)}\exp\left\lbrace -\frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace
\end{aligned}
$$

We group constants into a single coefficient, and drop them while maintaining proportionality.

$$
\begin{aligned}
p(\theta,\eta\mid y) &\propto \left(\frac{2}{\pi^2(2\pi)^{(1+T)/2}\prod_{t=1}^{T}y_t!}\right)\exp\left\lbrace \sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right)\right\rbrace \times \frac{\sqrt{1-\varphi^2}}{\sigma^{1+T}} \\
&\times \exp\left\lbrace -\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}(u_t - \varphi u_{t-1})^2\right)\right\rbrace \times \frac{1}{\left(1 + \sigma^2\right)}\exp\left\lbrace -\frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace \\
p(\theta,\eta\mid y) &\propto \frac{\sqrt{1-\varphi^2}}{\left(1+\sigma^2\right)\sigma^{1+T}}\exp\left\lbrace \sum_{t=1}^{T}\left(y_t\eta_t - e^{\eta_t}\right) -\frac{1}{2\sigma^2}\left(u_1^2\left(1-\varphi^2\right) + \sum_{t=2}^{T}\left(u_t - \varphi u_{t-1}\right)^2\right) - \frac{\beta_0^2}{2} - 2\beta_1^2\right\rbrace
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
