# Detailed Paper Outline: IR-Informed MPC and RL for Layerwise Overhang Thermal Control

## Proposed Title

**IR-Informed Predictive Thermal Control for Support-Free Overhangs in Material Extrusion Additive Manufacturing**

Alternative title:

**Layerwise Thermal-State Control of Material Extrusion Overhangs Using Infrared Sensing and Model Predictive Control**

---

## 1. Introduction

### 1.1 Motivation

Support-free overhangs are limited by a coupled thermal and mechanical trade-off:

- If the deposited material is too cold, interlayer bonding is poor.
- If the deposited material is too hot, the bead remains soft and can sag or collapse.
- Fixed slicer settings cannot adapt to local overhang angle, layer time, heat accumulation, or cooling variation.

The practical manufacturing problem is:

$$
\text{maximize support-free overhang quality}
$$

while satisfying:

$$
\text{sufficient bonding} \quad \text{and} \quad \text{geometric stability}
$$

### 1.2 Prior Work Positioning

The literature already includes:

- Machine learning for additive manufacturing process optimization.
- Reinforcement learning for additive manufacturing process control.
- Model predictive control and reduced-order models for additive manufacturing thermal or geometric regulation.
- Infrared thermography for temperature monitoring.
- Overhang printability studies in material extrusion.

Your seed bibliography supports this broader positioning, including reinforcement learning for additive manufacturing, reinforcement-learning-based MPC, direct ink writing control, wire arc additive manufacturing control, material-extrusion temperature regulation, and machine-learning-based additive manufacturing process optimization.

### 1.3 Gap

Prior work has not clearly demonstrated:

$$
\text{IR thermal feedback}
+
\text{layerwise predictive control}
+
\text{overhang-angle-dependent bonding window}
+
\text{support-free material extrusion outcome}
$$

The gap should be stated narrowly:

> Closed-loop control of the layerwise thermal state for support-free overhang formation in material extrusion remains underexplored.

### 1.4 Core Hypothesis

The paper hypothesis can be:

$$
\boxed{
\text{Regulating an IR-derived thermal-dose proxy within an overhang-dependent window improves support-free overhang fidelity.}
}
$$

More explicitly:

$$
D_{\min}(\alpha) \leq D_k \leq D_{\max}(\alpha)
$$

where:

- $D_k$ is the IR-derived thermal-dose proxy.
- $\alpha$ is the local overhang angle.
- $D_{\min}$ represents poor bonding risk.
- $D_{\max}$ represents sag or collapse risk.

### 1.5 Contributions

State three contributions:

1. **IR-derived layerwise thermal-state representation** for overhang printing.
2. **Reduced-order analytical thermal model** suitable for real-time MPC.
3. **Experimental comparison** of fixed settings, open-loop scheduling, MPC, and optional RL using the same IR feedback and overhang benchmark.

---

## 2. Experimental System

### 2.1 Printer and Material

Specify:

- Printer model.
- Nozzle diameter.
- Layer height.
- Filament or pellet material.
- Nozzle temperature range.
- Bed temperature.
- Fan configuration.
- Ambient or chamber temperature.
- Nominal print speed.

Example variables:

$$
d_n = \text{nozzle diameter}
$$

$$
h = \text{layer height}
$$

$$
w = \text{extrusion bead width}
$$

$$
T_{\infty} = \text{ambient temperature}
$$

$$
T_{\mathrm{nozzle}} = \text{nozzle setpoint}
$$

### 2.2 Overhang Benchmark Geometry

Use either discrete-angle coupons or a continuous ramp.

#### Option A: Discrete Overhang Coupon

Angles:

$$
\alpha \in \{45^\circ, 50^\circ, 55^\circ, 60^\circ, 65^\circ\}
$$

Metric:

$$
\alpha_{\max}
=
\max \{\alpha : \text{print is successful}\}
$$

#### Option B: Ramp Coupon

The local overhang angle varies continuously:

$$
\alpha(s) = \alpha_0 + \frac{\alpha_f-\alpha_0}{L}s
$$

where:

- $s$ is distance along the ramp.
- $L$ is total ramp length.
- $\alpha_0$ is initial angle.
- $\alpha_f$ is final angle.

This is good for one visual figure, but discrete coupons are better for statistics.

### 2.3 IR Thermal Sensing

Define the measured IR signal:

$$
y_k = T_{\mathrm{IR},k} + \epsilon_k
$$

where:

- $T_{\mathrm{IR},k}$ is the measured surface temperature.
- $\epsilon_k$ is measurement noise.

Choose a region of interest near the active overhang edge.

Possible IR features:

$$
T_{\mathrm{mean},k}
=
\frac{1}{|\Omega_k|}
\int_{\Omega_k} T_{\mathrm{IR}}(x,y,t_k)\,d\Omega
$$

$$
T_{\max,k}
=
\max_{(x,y)\in\Omega_k} T_{\mathrm{IR}}(x,y,t_k)
$$

$$
T_{\mathrm{edge},k}
=
\frac{1}{|\Omega_{\mathrm{edge},k}|}
\int_{\Omega_{\mathrm{edge},k}} T_{\mathrm{IR}}(x,y,t_k)\,d\Omega
$$

$$
\dot{T}_k
=
\frac{T_{\mathrm{IR},k}-T_{\mathrm{IR},k-1}}{t_k-t_{k-1}}
$$

Recommended main feedback variable:

$$
T_k = T_{\mathrm{edge},k}
$$

or:

$$
T_k = T_{\mathrm{mean},k}
$$

Use edge temperature if overhang sag is the main concern.

---

## 3. Thermal-State Definition

### 3.1 Why Not Control Nozzle Temperature Directly?

Nozzle temperature is an input, not the process state. The relevant process state is the thermal condition of the newly deposited bead and the bonding interface.

Because IR measures surface temperature, be precise:

$$
T_{\mathrm{IR},k}
\neq
T_{\mathrm{interface},k}
$$

Therefore, define an experimentally honest proxy:

$$
D_k = \text{IR-derived thermal-dose proxy}
$$

### 3.2 Thermal-Dose Proxy

Define:

$$
D_{k+1}
=
D_k
+
\Delta t_k
\max(T_k - T_g,0)
$$

where:

- $D_k$ is thermal dose.
- $T_k$ is the IR-measured layer temperature feature.
- $T_g$ is glass transition temperature or selected mobility threshold.
- $\Delta t_k$ is the time interval between decisions.

For semicrystalline polymers, replace $T_g$ with a suitable mobility or crystallization-related threshold if justified.

A more general form is:

$$
D_{k+1}
=
D_k
+
\Delta t_k
\left[
\frac{T_k-T_{\mathrm{ref}}}{T_{\mathrm{scale}}}
\right]_+
$$

where:

$$
[z]_+ = \max(z,0)
$$

This normalized dose may be easier if the true bonding threshold is uncertain.

### 3.3 Overhang-Dependent Thermal Window

Define:

$$
D_{\min}(\alpha)
\leq
D_k
\leq
D_{\max}(\alpha)
$$

Interpretation:

- $D_k < D_{\min}(\alpha)$: insufficient bonding.
- $D_k > D_{\max}(\alpha)$: excessive heat retention and sag risk.

Possible simple linear parameterization:

$$
D^\star(\alpha)
=
D_0
-
c_D(\alpha-\alpha_0)
$$

$$
D_{\min}(\alpha)
=
D^\star(\alpha) - \Delta D
$$

$$
D_{\max}(\alpha)
=
D^\star(\alpha) + \Delta D
$$

where:

- $D_0$ is the target dose at reference angle $\alpha_0$.
- $c_D$ controls how aggressively the thermal target decreases at larger overhang angles.
- $\Delta D$ is the allowed window width.

If you want a more flexible version:

$$
D^\star(\alpha)
=
b_0+b_1\alpha+b_2\alpha^2
$$

Start linear unless data justify nonlinear fitting.

---

## 4. Analytical Thermal Model for MPC

### 4.1 Model Purpose

The model only needs to predict temperature over a short horizon because IR feedback corrects the state after each layer or segment.

State this clearly:

> The reduced-order model is not intended to resolve the full three-dimensional transient thermal field. It is used for short-horizon control prediction and is corrected by IR measurements at each decision step.

### 4.2 One-State Model

Let:

$$
x_k = T_k
$$

The continuous lumped thermal model is:

$$
C\frac{dT}{dt}
=
Q_{\mathrm{in}}
-
H(T-T_{\infty})
$$

where:

- $C$ is effective thermal capacitance.
- $Q_{\mathrm{in}}$ is effective thermal input from deposition.
- $H$ is effective heat-loss coefficient.
- $T_{\infty}$ is ambient temperature.

The discrete analytical update is:

$$
T_{k+1}
=
T_{\infty}
+
(T_k^+ - T_{\infty})
\exp\left(-\frac{\Delta t_k}{\tau_k}\right)
$$

where:

$$
\tau_k = \frac{C}{H_k}
$$

and:

$$
T_k^+
=
(1-\beta)T_k+\beta T_{\mathrm{dep},k}
$$

### 4.3 Deposition Temperature Model

Approximate effective deposited temperature as:

$$
T_{\mathrm{dep},k}
=
T_{\mathrm{nozzle}}
-
a_v v_k
$$

where:

- $v_k$ is print speed.
- $a_v$ captures speed-dependent reduction in effective heat transfer.

If nozzle temperature is also controlled:

$$
T_{\mathrm{dep},k}
=
a_0+a_T T_{\mathrm{nozzle},k}-a_v v_k
$$

### 4.4 Cooling Time and Cooling Coefficient

Segment time:

$$
\Delta t_k
=
\frac{L_k}{v_k}+\tau_{\mathrm{dwell},k}
$$

Effective cooling time constant:

$$
\tau_k
=
\frac{\tau_0}
{1+a_f f_k+a_{\alpha}\frac{\alpha_k}{90}}
$$

where:

- $f_k$ is normalized fan speed.
- $\alpha_k$ is local overhang angle.
- $a_f$ captures fan-enhanced cooling.
- $a_{\alpha}$ captures geometry-dependent cooling or contact effects.

### 4.5 Recommended Model for Paper

Use:

$$
\boxed{
T_{k+1}
=
T_{\infty}
+
\left[
(1-\beta)T_k
+
\beta(T_{\mathrm{nozzle}}-a_v v_k)
-
T_{\infty}
\right]
\exp
\left(
-\frac{L_k/v_k+\tau_{\mathrm{dwell},k}}
{\tau_0/(1+a_f f_k+a_{\alpha}\alpha_k/90)}
\right)
}
$$

and:

$$
\boxed{
D_{k+1}
=
D_k
+
\Delta t_k
\max(T_{k+1}-T_g,0)
}
$$

This is compact, identifiable, and suitable for MPC.

---

## 5. Model Calibration

### 5.1 Calibration Data

Run calibration prints varying:

$$
v \in \{v_1,v_2,v_3\}
$$

$$
f \in \{f_1,f_2,f_3\}
$$

$$
\tau_{\mathrm{dwell}} \in \{\tau_1,\tau_2,\tau_3\}
$$

$$
\alpha \in \{\alpha_1,\alpha_2,\alpha_3\}
$$

Use IR temperature measurements at each segment or layer.

### 5.2 Parameter Vector

Fit:

$$
\theta =
[
\beta,
a_v,
\tau_0,
a_f,
a_{\alpha}
]
$$

Optional if nozzle temperature varies:

$$
\theta =
[
\beta,
a_0,
a_T,
a_v,
\tau_0,
a_f,
a_{\alpha}
]
$$

### 5.3 Least-Squares Objective

$$
\hat{\theta}
=
\arg\min_{\theta}
\sum_{k=1}^{M}
\left[
T_{k+1}^{\mathrm{IR}}
-
\hat{T}_{k+1}(\theta)
\right]^2
$$

Add bounds to keep parameters physically reasonable:

$$
0 < \beta < 1
$$

$$
\tau_0 > 0
$$

$$
a_f \geq 0
$$

If $a_{\alpha}$ is uncertain, allow it to be positive or negative and interpret carefully.

### 5.4 Model Validation Metrics

Report:

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{M}
\sum_{k=1}^{M}
\left(
T_{k}^{\mathrm{IR}}-\hat{T}_{k}
\right)^2
}
$$

$$
R^2
=
1-
\frac{
\sum_k (T_k^{\mathrm{IR}}-\hat{T}_k)^2
}{
\sum_k (T_k^{\mathrm{IR}}-\bar{T}^{\mathrm{IR}})^2
}
$$

Also report short-horizon prediction error:

$$
e_{N}
=
\frac{1}{N}
\sum_{i=1}^{N}
|T_{k+i}^{\mathrm{IR}}-\hat{T}_{k+i}|
$$

This matters more than long-horizon prediction for MPC.

---

## 6. MPC Formulation

### 6.1 State Vector

Use:

$$
x_k =
[
T_k,
D_k,
\alpha_k,
u_{k-1}
]^T
$$

Optional expanded state:

$$
x_k =
[
T_{\mathrm{mean},k},
T_{\mathrm{edge},k},
\dot{T}_k,
D_k,
\alpha_k,
l_k,
u_{k-1}
]^T
$$

For a short paper, keep the state simple.

### 6.2 Control Inputs

Recommended:

$$
u_k =
[
v_k,
f_k,
\tau_{\mathrm{dwell},k}
]^T
$$

where:

- $v_k$ is print speed.
- $f_k$ is fan speed.
- $\tau_{\mathrm{dwell},k}$ is dwell time.

Optional slow input:

$$
T_{\mathrm{nozzle},k}
$$

But for real-time control, nozzle temperature may be too slow. It is safer to use nozzle temperature as a layer-level supervisory variable rather than a segment-level actuator.

### 6.3 Prediction Model

$$
x_{k+1}
=
F(x_k,u_k,\alpha_k)
$$

where:

$$
T_{k+1}=f_T(T_k,u_k,\alpha_k)
$$

$$
D_{k+1}=D_k+\Delta t_k\max(T_{k+1}-T_g,0)
$$

### 6.4 Objective Function

$$
\boxed{
J_{\mathrm{MPC}}
=
\sum_{i=0}^{N-1}
\left[
w_T
(T_{k+i}-T^\star(\alpha_{k+i}))^2
+
w_D
(D_{k+i}-D^\star(\alpha_{k+i}))^2
+
w_u
\|u_{k+i}-u_{k+i-1}\|_2^2
+
w_{\tau}\tau_{\mathrm{dwell},k+i}
\right]
}
$$

Optional sag penalty:

$$
+
w_s \hat{s}_{k+i}^2
$$

where $\hat{s}_{k+i}$ is predicted sag.

If you do not have a sag model, keep sag as an experimental outcome, not an MPC state.

### 6.5 Constraints

Actuator constraints:

$$
v_{\min}\leq v_k\leq v_{\max}
$$

$$
0\leq f_k\leq 1
$$

$$
0\leq \tau_{\mathrm{dwell},k}\leq \tau_{\max}
$$

Rate constraints:

$$
|v_k-v_{k-1}|\leq \Delta v_{\max}
$$

$$
|f_k-f_{k-1}|\leq \Delta f_{\max}
$$

Thermal constraints:

$$
D_{\min}(\alpha_k)\leq D_k\leq D_{\max}(\alpha_k)
$$

$$
T_k\leq T_{\mathrm{sag}}(\alpha_k)
$$

Optional lower temperature constraint:

$$
T_k\geq T_{\mathrm{bond}}(\alpha_k)
$$

### 6.6 MPC Decision Rule

At each layer or segment:

$$
u_k^\star
=
\arg\min_{u_{k:k+N-1}}
J_{\mathrm{MPC}}
$$

Only apply the first action:

$$
u_k = u_k^\star(0)
$$

Then update:

$$
T_k \leftarrow T_{\mathrm{IR},k}
$$

$$
D_k \leftarrow D_k+\Delta t_k\max(T_{\mathrm{IR},k}-T_g,0)
$$

Then solve again.

### 6.7 MPC Implementation Details

Recommended settings:

$$
N = 5 \text{ to } 10
$$

Decision interval:

- Segment-level control if your printer accepts real-time G-code changes.
- Layer-level control if communication is slower.

Optimization method:

- Sequential quadratic programming.
- Interior-point method.
- SciPy SLSQP for prototyping.

Warm start:

$$
u_{k:k+N-1}^{(0)}
=
[u_{k-1},u_{k-1},...,u_{k-1}]
$$

Fallback action if optimizer fails:

$$
u_k = u_{k-1}
$$

or conservative cooling action:

$$
u_k = [v_{\mathrm{low}}, f_{\mathrm{high}}, \tau_{\mathrm{moderate}}]
$$

---

## 7. Reinforcement Learning Formulation

### 7.1 Purpose of RL in This Paper

RL should be a comparator or extension, not necessarily the main claim.

Use RL to answer:

> Can a learned policy using the same IR thermal state outperform or match MPC without solving an online optimization problem?

### 7.2 State

Use the same state as MPC:

$$
s_k =
[
T_k,
D_k,
\dot{T}_k,
\alpha_k,
u_{k-1}
]^T
$$

Optional:

$$
s_k =
[
T_{\mathrm{mean},k},
T_{\mathrm{max},k},
T_{\mathrm{edge},k},
D_k,
\dot{T}_k,
\alpha_k,
l_k,
v_{k-1},
f_{k-1},
\tau_{k-1}
]^T
$$

### 7.3 Action

Continuous action:

$$
a_k =
[
v_k,
f_k,
\tau_{\mathrm{dwell},k}
]^T
$$

Use SAC or TD3 for continuous control.

Discrete action alternative:

$$
v_k \in \{20,30,40,50\}
$$

$$
f_k \in \{0,0.5,1.0\}
$$

$$
\tau_k \in \{0,1,3,5\}
$$

Use DQN only if actions are discrete.

### 7.4 Reward

Define reward as negative thermal-control cost:

$$
r_k
=
-
\left[
w_T
(T_k-T^\star(\alpha_k))^2
+
w_D
(D_k-D^\star(\alpha_k))^2
+
w_u
\|u_k-u_{k-1}\|_2^2
+
w_{\tau}\tau_k
+
P_k
\right]
$$

Thermal-window penalty:

$$
P_k
=
\lambda_{\min}
[D_{\min}(\alpha_k)-D_k]_+^2
+
\lambda_{\max}
[D_k-D_{\max}(\alpha_k)]_+^2
$$

Failure penalty:

$$
P_{\mathrm{fail},k}
=
\begin{cases}
P_f, & \text{if severe sag, collapse, or delamination occurs} \\
0, & \text{otherwise}
\end{cases}
$$

Total reward:

$$
r_k = r_{\mathrm{thermal},k} - P_{\mathrm{fail},k}
$$

### 7.5 RL Objective

$$
\pi^\star
=
\arg\max_{\pi}
\mathbb{E}_{\pi}
\left[
\sum_{k=0}^{K}
\gamma^k r_k
\right]
$$

where:

- $\pi$ is the policy.
- $\gamma$ is the discount factor.
- $K$ is episode length.

### 7.6 RL Training Environment

Three options:

#### Option 1: Train on Analytical Model

Use the same thermal model as MPC.

Advantage:

- Fast.
- Safe.
- Easy comparison.

Weakness:

- RL may learn model artifacts.

#### Option 2: Train on Calibrated Digital Twin

Add stochasticity:

$$
T_{k+1}=f_T(T_k,u_k,\alpha_k)+\eta_k
$$

$$
\eta_k \sim \mathcal{N}(0,\sigma_T^2)
$$

Randomize parameters:

$$
\theta \sim \mathcal{U}(\theta_{\min},\theta_{\max})
$$

Advantage:

- More robust policy.
- Better simulation-to-experiment transfer.

#### Option 3: Train Physically

Not recommended for initial letter unless experiments are safe and cheap.

### 7.7 Safety Layer for RL

To make RL experimentally acceptable, use action clipping and thermal constraints:

$$
u_k = \mathrm{clip}(\pi(s_k),u_{\min},u_{\max})
$$

Optional MPC safety filter:

$$
u_k =
\arg\min_{u}
\|u-\pi(s_k)\|^2
$$

subject to:

$$
D_{\min}(\alpha_k)\leq \hat{D}_{k+1}\leq D_{\max}(\alpha_k)
$$

This creates a safe RL or RL-MPC hybrid.

---

## 8. Baseline Controllers

Use four controllers if space allows.

### 8.1 Fixed Slicer Baseline

$$
u_k = u_0
$$

where:

$$
u_0=[v_0,f_0,\tau_0]
$$

This represents conventional printing.

### 8.2 Open-Loop Angle Schedule

$$
u_k = g(\alpha_k)
$$

Example:

$$
v_k = v_0 - c_v(\alpha_k-\alpha_0)
$$

$$
f_k = f_0 + c_f(\alpha_k-\alpha_0)
$$

$$
\tau_k = \tau_0 + c_{\tau}(\alpha_k-\alpha_0)
$$

This baseline tests whether feedback matters beyond simple geometry scheduling.

### 8.3 MPC

$$
u_k =
\arg\min J_{\mathrm{MPC}}
$$

Uses IR feedback and thermal prediction.

### 8.4 RL

$$
u_k=\pi_{\phi}(s_k)
$$

Uses IR feedback but no online optimization.

---

## 9. Evaluation Metrics

### 9.1 Thermal Tracking Error

$$
E_T
=
\frac{1}{K}
\sum_{k=1}^{K}
|T_k-T^\star(\alpha_k)|
$$

### 9.2 Thermal-Dose Tracking Error

$$
E_D
=
\frac{1}{K}
\sum_{k=1}^{K}
|D_k-D^\star(\alpha_k)|
$$

### 9.3 Thermal-Window Violation Rate

$$
V_D
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbb{1}
\left[
D_k<D_{\min}(\alpha_k)
\text{ or }
D_k>D_{\max}(\alpha_k)
\right]
$$

### 9.4 Sag Metric

If you scan or photograph the overhang:

$$
s_k =
z_{\mathrm{nominal},k}-z_{\mathrm{measured},k}
$$

Mean sag:

$$
\bar{s}
=
\frac{1}{K}
\sum_{k=1}^{K}
s_k
$$

Peak sag:

$$
s_{\max}
=
\max_k s_k
$$

### 9.5 Maximum Printable Overhang Angle

$$
\alpha_{\max}
=
\max
\{
\alpha:
s(\alpha)\leq s_{\mathrm{tol}}
\text{ and no collapse occurs}
\}
$$

where $s_{\mathrm{tol}}$ is an allowable sag threshold.

### 9.6 Print Time

$$
t_{\mathrm{print}}
=
\sum_{k=1}^{K}
\left(
\frac{L_k}{v_k}+\tau_k
\right)
$$

This prevents the controller from winning only by adding excessive dwell.

### 9.7 Control Smoothness

$$
TV_u
=
\sum_{k=1}^{K}
\|u_k-u_{k-1}\|_2
$$

This quantifies whether control actions are practical.

### 9.8 Optional Bond-Strength Metric

For mechanical validation:

$$
F_{\mathrm{fail}}
=
\text{failure load during bending, peel, or tensile test}
$$

or normalized strength:

$$
\sigma_{\mathrm{norm}}
=
\frac{\sigma_{\mathrm{controlled}}}{\sigma_{\mathrm{baseline}}}
$$

---

## 10. Experimental Matrix

### 10.1 Minimum Matrix for a Short Letter

| Controller | 45° | 55° | 65° | Ramp coupon |
|---|---:|---:|---:|---:|
| Fixed | 3 | 3 | 3 | 1 |
| Open-loop | 3 | 3 | 3 | 1 |
| MPC | 3 | 3 | 3 | 1 |
| RL | optional | optional | optional | optional |

If RL is not mature, move it to simulation or supplementary results.

### 10.2 Recommended Primary Comparison

Use:

$$
\text{fixed}
\quad
\text{vs}
\quad
\text{open-loop}
\quad
\text{vs}
\quad
\text{MPC}
$$

Add RL only if:

- Training is complete.
- Policy is stable.
- Constraint violations are acceptable.
- The comparison does not distract from the main contribution.

---

## 11. Figure Plan for Four-Figure Limit

### Figure 1: System Concept and Control Architecture

Include:

- Printer nozzle.
- Overhang coupon.
- IR sensor field of view.
- Region of interest.
- Controller.
- Inputs $v_k$, $f_k$, $\tau_k$.
- Output $T_{\mathrm{IR},k}$.
- Thermal-dose window.

Main message:

$$
\text{IR feedback enables overhang-aware thermal control.}
$$

### Figure 2: Thermal Model and Calibration

Panels:

A. Representative IR thermal image or temperature map.  
B. Cooling curves from calibration prints.  
C. Model prediction versus measured IR temperature.  
D. Thermal-dose calculation and target window.

Main message:

$$
\text{A low-order model is sufficiently predictive over the MPC horizon.}
$$

### Figure 3: Closed-Loop Thermal Control

Panels:

A. $D_k$ versus $D_{\min}$, $D^\star$, and $D_{\max}$.  
B. $T_k$ versus $T^\star(\alpha)$.  
C. Control actions $v_k$, $f_k$, $\tau_k$.  
D. Thermal-window violation rate across controllers.

Main message:

$$
\text{MPC reduces thermal-window violations compared with fixed and open-loop control.}
$$

### Figure 4: Manufacturing Outcome

Panels:

A. Photographs or 3D scans of overhang coupons.  
B. Sag versus overhang angle.  
C. Maximum successful overhang angle.  
D. Print time or bond-strength comparison.

Main message:

$$
\text{Thermal regulation improves support-free overhang quality.}
$$

---

## 12. Results Section Outline

### 12.1 Model Calibration Result

Claim to support:

> The reduced-order model captured the dominant cooling dynamics over the MPC prediction horizon.

Report:

$$
\mathrm{RMSE}=[X]^\circ\mathrm{C}
$$

$$
R^2=[Y]
$$

$$
e_N=[Z]^\circ\mathrm{C}
$$

Possible discussion:

- Prediction was best for supported or moderate overhang regions.
- Error increased at high overhang angles because bead deformation and changing contact area are not explicitly modeled.
- Short-horizon correction with IR feedback reduced the impact of model mismatch.

### 12.2 Thermal Control Result

Claim to support:

> MPC kept the IR-derived thermal dose within the overhang-dependent window more consistently than fixed or open-loop control.

Report:

$$
V_D^{\mathrm{fixed}}=[A]\%
$$

$$
V_D^{\mathrm{open-loop}}=[B]\%
$$

$$
V_D^{\mathrm{MPC}}=[C]\%
$$

Possible discussion:

- Fixed settings overheated steep overhangs or underheated early layers.
- Open-loop scheduling improved average behavior but did not correct local variation.
- MPC adapted based on the measured thermal state.

### 12.3 Overhang Geometry Result

Claim to support:

> Reduced thermal-window violation translated into improved geometric fidelity.

Report:

$$
s_{\max}^{\mathrm{fixed}}=[A]\text{ mm}
$$

$$
s_{\max}^{\mathrm{MPC}}=[B]\text{ mm}
$$

$$
\alpha_{\max}^{\mathrm{fixed}}=[C]^\circ
$$

$$
\alpha_{\max}^{\mathrm{MPC}}=[D]^\circ
$$

Possible discussion:

- Improvement should be strongest near the failure threshold.
- At mild overhang angles, all controllers may perform similarly.
- At very steep angles, thermal control alone may not prevent failure because bead support and path geometry dominate.

### 12.4 RL Comparison Result

If RL is included:

Possible claim if RL performs well:

> RL achieved comparable thermal tracking after training but did not explicitly enforce constraints.

Possible claim if RL performs worse:

> MPC provided better constraint satisfaction and sample efficiency than RL, supporting its use for initial deployment in thermal overhang control.

Report:

$$
E_D^{\mathrm{RL}}
$$

$$
V_D^{\mathrm{RL}}
$$

$$
s_{\max}^{\mathrm{RL}}
$$

Also report training requirement:

$$
N_{\mathrm{episodes}}=[X]
$$

$$
N_{\mathrm{prints}}=[Y]
$$

This highlights the practical cost of RL.

---

## 13. Discussion Points

### 13.1 Why MPC May Be Preferable Initially

MPC is attractive because:

- It is interpretable.
- It handles constraints directly.
- It needs less training data.
- It can be updated using measured IR feedback.
- It is safer for physical experiments.

Phrase carefully:

> The results suggest that explicit prediction and constraint handling are advantageous for early-stage closed-loop thermal control in material extrusion.

### 13.2 Why RL May Still Be Useful

RL may be useful for:

- Learning $D^\star(\alpha)$.
- Learning the terminal value function for MPC.
- Learning residual model corrections.
- Handling nonlinear delayed sag effects.
- Improving policies after enough experimental data are collected.

Hybrid formulation:

$$
x_{k+1}
=
f_{\mathrm{analytical}}(x_k,u_k)
+
g_{\theta}(x_k,u_k)
$$

where $g_{\theta}$ is learned from IR data.

Or:

$$
J_{\mathrm{MPC}}
=
\sum_{i=0}^{N-1}\ell(x_i,u_i)
+
V_{\theta}(x_N)
$$

where $V_{\theta}$ is a learned terminal value.

### 13.3 Limitations

State limitations explicitly:

1. IR measures surface temperature, not buried interface temperature.
2. Thermal dose is a proxy, not a direct molecular diffusion measurement.
3. Sag is affected by bead geometry, viscoelasticity, cooling, toolpath, and flow rate.
4. The analytical model may not generalize across materials without recalibration.
5. Nozzle temperature may be too slow for segment-level control.
6. The proposed method does not replace mechanical support for extreme overhangs.

### 13.4 Practical Implication

The practical implication is:

$$
\text{closed-loop thermal state control}
\rightarrow
\text{reduced support dependence}
\rightarrow
\text{less post-processing and better geometric fidelity}
$$

Phrase:

> The proposed framework moves overhang printing from fixed slicer heuristics toward feedback-regulated thermal management.

---

## 14. Conclusion Outline

Include four points:

1. IR sensing provides a layerwise thermal feedback signal during overhang printing.
2. A reduced-order analytical model is sufficient for short-horizon MPC.
3. MPC regulates thermal dose within an overhang-dependent bonding window.
4. Improved thermal-window regulation is expected to reduce sag and increase support-free overhang printability.

Possible final sentence:

> The main contribution is not the use of IR sensing or MPC alone, but their integration into an overhang-aware thermal control framework that links measured layerwise temperature to support-free manufacturing quality.

---

## 15. Suggested Implementation Workflow

### Step 1: Sensor Calibration

- Calibrate IR emissivity for the printed material.
- Define the thermal region of interest.
- Validate IR temperature against a contact thermocouple or known reference if possible.

### Step 2: Calibration Prints

Print simple tracks while varying:

$$
v, f, \tau_{\mathrm{dwell}}, \alpha
$$

Collect:

$$
(T_k,u_k,\alpha_k,T_{k+1})
$$

### Step 3: Fit Thermal Model

Solve:

$$
\hat{\theta}
=
\arg\min_{\theta}
\sum_k
(T_{k+1}^{\mathrm{IR}}-\hat{T}_{k+1})^2
$$

### Step 4: Define Thermal Window

Empirically identify:

$$
D_{\min}(\alpha)
$$

from poor bonding cases.

Identify:

$$
D_{\max}(\alpha)
$$

from sag or collapse cases.

### Step 5: Implement MPC

At each segment or layer:

1. Read $T_{\mathrm{IR},k}$.
2. Compute $D_k$.
3. Read or compute future $\alpha_{k:k+N}$ from G-code.
4. Solve MPC.
5. Apply $v_k$, $f_k$, and $\tau_k$.
6. Repeat.

### Step 6: Print Benchmark Coupons

Run:

$$
\text{fixed}, \text{ open-loop}, \text{ MPC}
$$

Optional:

$$
\text{RL}
$$

### Step 7: Measure Outcomes

Measure:

$$
E_D, V_D, s_{\max}, \bar{s}, \alpha_{\max}, t_{\mathrm{print}}
$$

### Step 8: Prepare Four Figures

Use:

1. Concept.
2. Model calibration.
3. Thermal control.
4. Manufacturing outcome.

---

## 16. Most Defensible Novelty Statement

Use this nearly verbatim:

> This work introduces an IR-informed predictive control framework for support-free material extrusion overhangs. Unlike fixed slicer settings or open-loop angle schedules, the proposed method uses layerwise infrared measurements to regulate a thermal-dose proxy within an overhang-dependent bonding window. The approach explicitly balances interlayer bonding and sag resistance through constrained model predictive control.

This is the safest and strongest framing.
