#set text(font: "sans-serif")

= `cellular_raza` - Pool Model

#table(
    columns: (auto, auto, auto),
    inset: 10pt,
[], [== ABM], [== ODE],
[
    === Cell-Cycle
    1. Lag-phase
    2. Growth
    3. Division
],[],
[
    - Lag-Phase $lambda_1, lambda_2$
    - Growth Phase $alpha_1, alpha_2$
],
[
    === Intra- & Extracellular
    - Production
    - Secretion
    - Transport in Microenvironment (Diffusion $D_R D_I$)
],[],
[
    - Inhibition $mu_I, K$
    - Resource $N_t$
],
[
    === Physical Mechanics
    - Force Potential
    - Random motion?
],
)

== Cell-Cycle
Define the division threshold
$T = frac(R_e (t=0) "vol(Domain)", N_"max cells" "vol(cell)")$

In our case, we can savely assume $N_"max cells"=N_t$.

We transition with the probability $p=lambda Delta t$ from the lag-phase to the growth phase.
```rs
let q = rng.gen_range(0.0..1.0);
if q <= self.transition_rate * dt {
    self.is_in_lag_phase = false;
}
```
Afterwards, we divide regularly as long as enough nutrients are present

```rust
if self.intracellular_resource >= T {
    self.intracellular_resource -= T;
    self.divide();
}
```
== Intra- & Extracellular Reactions

$frac(diff  R_(i,k), diff t) &= alpha frac(R_e (x_k), 1 + mu_I I_e (x_k))\
frac(diff I_(i,k), diff t) &= 0\
frac(diff R_e, diff t) &= -frac(alpha, 1 + mu_I I_e)R_e sum_(k=0)^n delta(x_k) &+ D_R Delta R_e\
frac(diff I_e, diff t) &= underbracket(K, "not in lag phase") sum_(k=0)^n delta(x_k) &+ D_I Delta I_e$

=== Initial Conditions
Everything is $0$ except for the extracellular resource $R_e = frac(N_t, "vol(domain)")$

== Physical Mechancis
