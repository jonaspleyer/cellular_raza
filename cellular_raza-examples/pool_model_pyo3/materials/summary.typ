#set text(font: "sans-serif")

= `cellular_raza` - Pool Model

#table(
    columns: (auto, auto, auto),
    inset: 10pt,
[== ABM], [== ODE], [== Comments],
[
    === Cell-Cycle
    1. Lag-phase
    2. Growth
    3. Division
],[
    - Lag-Phase $lambda_1, lambda_2$
    - Growth Phase $alpha_1, alpha_2$
],[],
[
    === Intra- & Extracellular
    - Production
    - Secretion
    - Transport in Microenvironment (Diffusion $D_R D_w$)
],[
    - Inhibition $M omega_w, K$
    - Resource $N_t$
],[],
[
    === Physical Mechanics
    - Force Potential
    - Random motion?
],[],[]
)
