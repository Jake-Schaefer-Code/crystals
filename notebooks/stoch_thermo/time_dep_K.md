Now, consider a time-dependent rate matrix $K(\lambda_t)$, which is nonlinear in its time-dependence via $\lambda_t$ (as otherwise we'd write it $Kt$). Our master equation is
$$
\dot p_i(t) = \sum_{j} K_{ij}(\lambda_t) p_j(t)
$$
The solution to such a master equation is
$$
p_s = U(s, 0)p_0,\quad \partial_s U(s, 0)  = K(\lambda_s) U(s, 0),
$$
where
$$
U(s, 0) = \mathcal{T}\exp\left(\int_0^s K(\lambda_\tau)d\tau\right)
$$

$$
D(p_0\|q) - D(U(s, 0)p_0\| U(s, 0)q)
$$
Note:
$$
\frac{d}{ds} \left(\frac{p_i}{q_i}\right)
 = \frac{\dot p_i}{q_i} - \dot q_i \frac{p_i}{q_i^2} 
 = \frac{p_i}{q_i} \left(\frac{\dot p_i}{p_i} - \frac{\dot q_i }{q_i} \right)
$$
and
$$
\frac{d}{ds} \log\left(\frac{p_i}{q_i}\right) 
= \frac{q_i}{p_i} \frac{d}{ds} \left(\frac{p_i}{q_i}\right) 
= \frac{\dot p_i}{p_i} - \frac{\dot q_i }{q_i}
$$
Thus, 
$$
\frac{d}{ds} D(p_s\| q_s) 
= \sum_i \dot p_i \log\left(\frac{p_i}{q_i}\right)  + p_i\left(\frac{\dot p_i}{p_i} - \frac{\dot q_i }{q_i}\right)
$$
$$
=\sum_i \dot p_i \log\left(\frac{p_i}{q_i}\right)  + \sum_i \dot p_i - \sum_i \frac{p_i}{q_i}\dot q_i,
$$
but because $p_s$ is normalized, $\sum_i \dot p_i=0$, so 
$$
\frac{d}{ds} D(p_s\| q_s) = \dot p_s^\top \log\left(\frac{p_s}{q_s}\right) -  \left(\frac{p_s}{q_s}\right)^\top\dot q_s.
$$
Now, if both distributions satisfy the master equation, so 
$$
\dot p_s = K(\lambda_s)p_s,\quad\dot q_s = K(\lambda_s)q_s,
$$
giving us 
$$
\frac{d}{ds} D(p_s\| q_s) 
= (K(\lambda_s)p_s)^\top\log\left(\frac{p_s}{q_s}\right) -  \left(\frac{p_s}{q_s}\right)^\top(K(\lambda_s)q_s).
$$
Let
$$
r_i = \frac{p_i}{q_i},
$$
so

$$
\begin{align}
\frac{d}{ds} D(p_s\| q_s) 
&= \sum_i (K(\lambda_s)p_s)_i\log r_i - \sum_i r_i(K(\lambda_s)q_s)_i
\end{align} 
$$
so the first term follows
$$
\begin{align}
\sum_i (K(\lambda_s)p_s)_i\log r_i  &= \sum_i \log r_i \sum_j K_{ij}(\lambda_s)q_jr_j \\

&= \sum_i \log r_i \left(\sum_{j\neq i} K_{ij}(\lambda_s)q_jr_j + K_{ii}(\lambda_s)q_ir_i\right) \\

&= \sum_i \log r_i \left(\sum_{j\neq i} K_{ij}(\lambda_s)q_jr_j - q_ir_i\sum_{j\neq i}K_{ji}(\lambda_s)\right) \\

&= \sum_i \sum_{j\neq i} K_{ij}q_j(r_j\log r_i) - \sum_i \sum_{j\neq i} K_{ji} q_i (r_i\log r_i) \\

&= \sum_i \sum_{j\neq i} K_{ij}q_j(r_j\log r_i) - \sum_j \sum_{i\neq j} K_{ij} q_j (r_j\log r_j) \\

&= \sum_{i\neq j} K_{ij}q_j r_j\log\left(\frac{r_i}{r_j}\right) 
\end{align} 
$$


And the second term,
$$
\begin{align}
  \sum_i r_i(K(\lambda_s)q_s)_i 
  &= \sum_i r_i \sum_j K_{ij}(\lambda_s) q_j \\
  &= \sum_i r_i \left(\sum_{j\neq i} K_{ij}(\lambda_s) q_j + K_{ii} q_i\right)\\
  &= \sum_i  \sum_{j\neq i} K_{ij}(\lambda_s) q_j (r_i) - \sum_i  \sum_{j\neq i} K_{ji} q_i (r_i)  \\
  &= \sum_i  \sum_{j\neq i} K_{ij}(\lambda_s) q_j (r_i) - \sum_j  \sum_{i\neq j} K_{ij} q_j (r_j)  \\
  &= \sum_{i\neq j}  K_{ij}(\lambda_s) q_j (r_i - r_j) 
\end{align}
$$


Thus,
$$
\begin{align}
\frac{d}{ds} D(p_s\| q_s) 
&= \sum_{i\neq j} K_{ij}q_j r_j\log\left(\frac{r_i}{r_j}\right) - \sum_{i\neq j}  K_{ij}(\lambda_s) q_j (r_i - r_j) \\
&= \sum_{i\neq j}  K_{ij}(\lambda_s) q_j\left(r_j\log\left(\frac{r_i}{r_j}\right) - r_i +r_j \right) \\
&= -\sum_{i\neq j}  K_{ij}(\lambda_s) q_j\left(r_i - r_j - r_j\log\left(\frac{r_i}{r_j}\right) \right),
\end{align} 
$$
and since 
$$
r_i - r_j - r_j\log\left(\frac{r_i}{r_j}\right) \geq 0, \quad K_{ij}\geq 0, \, i\neq j,
$$
we have
$$
\frac{d}{ds} D(p_s\| q_s) \leq 0 \implies - \frac{d}{ds} D(p_s\| q_s) \geq 0
$$

In particular, plutgging back in $r_i = p_i/q_i$,
$$
\begin{align}
\frac{d}{ds} D(p_s\| q_s) 
&= -\sum_{i\neq j}  K_{ij}(\lambda_s) q_j\left(\frac{p_i}{q_i} - \frac{p_j}{q_j} - \frac{p_j}{q_j}\log\left(\frac{p_iq_j}{p_jq_i}\right) \right) \\
&= -\sum_{i\neq j}  K_{ij}(\lambda_s)\left(q_j \frac{p_i}{q_i} - p_j - p_j\log\left(\frac{p_iq_j}{p_jq_i}\right) \right) \\
\end{align} 
$$

Now,

$$
\begin{align}
    \Delta\mathcal{M}_T&:=\mathcal{M}_T(q)-\mathcal{M}_T(\bar q)\nonumber\\
    &=\int_0^T D_K(p_t\|q) - D_K(p_t\|\bar q)dt.
\end{align}
$$


and

$$
\begin{align}
  D_K(p_t\|q) - D_K(p_t\|\bar q) 
  &= -\sum_{i\neq j}  K_{ij}(\lambda_t)\left(q_j \frac{p_i}{q_i} - p_j - p_j\log\left(\frac{p_iq_j}{p_jq_i}\right) \right) + \sum_{i\neq j}  K_{ij}(\lambda_t)\left(\bar q_j \frac{p_i}{\bar q_i} - p_j - p_j\log\left(\frac{p_i\bar q_j}{p_j\bar q_i}\right) \right) \\

  &=  \sum_{i\neq j}  K_{ij}(\lambda_t)\left[\bar q_j \frac{p_i}{\bar q_i} - p_j - p_j\log\left(\frac{p_i\bar q_j}{p_j\bar q_i}\right)  - q_j \frac{p_i}{q_i} + p_j + p_j\log\left(\frac{p_iq_j}{p_jq_i}\right) \right]  \\
  &= \sum_{i\neq j}  K_{ij}(\lambda_t)\left[\bar q_j \frac{p_i}{\bar q_i}  - q_j \frac{p_i}{q_i}  + p_j\log\left( \frac{p_iq_j}{p_jq_i}  \frac{p_j\bar q_i}{p_i\bar q_j} \right) \right]  \\
  &= \sum_{i\neq j}  K_{ij}(\lambda_t)\left[\bar q_j \frac{p_i}{\bar q_i}  - q_j \frac{p_i}{q_i}  + p_j\log\left( \frac{q_j}{q_i}  \frac{\bar q_i}{\bar q_j} \right) \right]  \\

  &= \sum_{i\neq j} \frac{p_i}{\bar q_i} K_{ij}(\lambda_t) \bar q_j - \sum_{i\neq j} \frac{p_i}{q_i} K_{ij}(\lambda_t) q_j + \sum_{i\neq j} K_{ij}(\lambda_t) \left[ p_j\log\left( \frac{q_j}{q_i}  \frac{\bar q_i}{\bar q_j} \right) \right]  \\

  &=  \left(\frac{p_t}{\bar q}\right)^\top (K(\lambda_t) \bar q)
    - \sum_{i} \frac{p_i}{\bar q_i} K_{ii}(\lambda_t) \bar q_i -\left(\frac{p_t}{q}\right)^\top (K(\lambda_t) q )
   + \sum_{i} \frac{p_i}{q_i} K_{ii}(\lambda_t) q_i \\
   &+ \sum_{i\neq j} K_{ij}(\lambda_t) p_j \log\frac{\bar q_i}{q_i} - \sum_{i\neq j} K_{ij}(\lambda_t) p_j\log \frac{\bar q_j}{ q_j}  \\
  
  &= \left(\frac{p_t}{\bar q}\right)^\top (K(\lambda_t) \bar q) -\left(\frac{p_t}{q}\right)^\top (K(\lambda_t) q )
  + \sum_{i\neq j} K_{ij}(\lambda_t) p_j \log\frac{\bar q_i}{q_i} + \sum_{i} K_{ii}(\lambda_t) p_i\log \frac{\bar q_i}{ q_i} \\

  &= \left(\frac{p_t}{\bar q}\right)^\top (K(\lambda_t) \bar q) -\left(\frac{p_t}{q}\right)^\top (K(\lambda_t) q )
  + (K(\lambda_t)p)^\top \log\left(\frac{\bar q}{q}\right)

\end{align}
$$



Thus, 
$$
\begin{align}
    \int_0^T D_K(p_t\|q) - D_K(p_t\|\bar q)dt &= \int_0^T \left(\frac{p_t}{\bar q}\right)^\top (K(\lambda_t) \bar q)dt - \int_0^T  \left(\frac{p_t}{q}\right)^\top (K(\lambda_t) q )dt
  + \int_0^T  (K(\lambda_t)p_t)^\top \log\left(\frac{\bar q}{q}\right) dt \\
\end{align}
$$
but
$$
\int_0^T  (K(\lambda_t)p_t)^\top \log\left(\frac{\bar q}{q}\right) dt = \log\left(\frac{\bar q}{q}\right)^\top \int_0^T \dot p_t dt = \log\left(\frac{\bar q}{q}\right)^\top (p_T - p_0),
$$

so we have 
$$
\begin{align}
    \int_0^T D_K(p_t\|q) - D_K(p_t\|\bar q)dt 
    &= \int_0^T \left(\frac{p_t}{\bar q}\right)^\top (K(\lambda_t) \bar q)dt - \int_0^T  \left(\frac{p_t}{q}\right)^\top (K(\lambda_t) q )dt \\
    &+(p_T - p_0)^\top \log\left(\frac{\bar q}{q}\right) \\
\end{align}
$$

Also, note, continuous time telescope:
$$
p_T - p_0 = \int_0^T \dot p_t dt = \int_0^T K p_t dt,
$$
so
$$
\begin{align}
p_T - p_0 
&= \int_0^T K(0)p_t dt + \int_0^T K'(0)t p_t dt + \int_0^T \frac{1}{2}K''(0)t^2p_t dt + \cdots \\
&= TK\bar q + K'(0) \int_0^T t p_t dt + \frac{K''(0)}{2}\int_0^T t^2p_t dt + \cdots \\
\end{align}
$$