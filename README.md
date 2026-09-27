# Modern Integrated Navigation Systems

## 3D RISS Mechanization, Robust Single-Point Positioning, and Loosely Coupled GNSS/INS Fusion

**Author:** Mazen Elaraby  
**Affiliation:** Department of Electrical and Computer Engineering, Queen's University, Kingston, Canada

This repository develops a complete land-vehicle navigation pipeline in three stages:

1. **3D Reduced Inertial Sensor System (RISS) mechanization** in the local-level frame (LLF), using one vertical gyroscope, three accelerometers, and OBD-II odometry;
2. **Single-Point GPS Positioning (SPP)** from raw pseudorange and Doppler measurements, with particular emphasis on ill-conditioning, robust matrix factorization, elevation weighting, and DOP-based quality control; and
3. **Loosely coupled GNSS/3D RISS integration** using a nine-state error-state Kalman filter, including first-order Gauss-Markov sensor-error models and simulated GNSS outages.

The experiments use a synchronized, approximately **12.6 km / 42 min** road trajectory. The repository compares tactical/high-end and automotive MEMS inertial data, studies the numerical failure modes of GNSS least squares, and quantifies dead-reckoning drift when absolute updates are unavailable.

> **Sensor naming note.** The project report context refers to the tactical-grade unit as a Honeywell HG1700. The checked-in dataset documentation and code identify the high-end stream as **NovAtel/KVH-1750** (`high_end_imu`, `kvh_*`). The low-end stream is consistently identified as the **VTI SCC1300-D04**. This README retains the project-level “tactical/high-end” comparison while using the repository labels when referring to files or parameters.

---

## Contents

- [System architecture](#system-architecture)
- [Module 1 — 3D RISS mechanization](#module-1--3d-riss-mechanization-in-llf)
- [Module 2 — Single-Point GPS Positioning](#module-2--single-point-gps-positioning)
- [Module 3 — Loosely coupled GNSS/3D RISS fusion](#module-3--loosely-coupled-gnss3d-riss-fusion)
- [Performance benchmarks](#performance-benchmarks-and-key-findings)
- [Repository structure](#repository-structure)
- [Setup and reproduction](#setup-and-how-to-run)
- [Implementation notes](#implementation-notes-and-reproducibility-caveats)

---

## System Architecture

```text
 Raw sensor logs                     Raw GNSS observables
 ┌───────────────────────┐          ┌──────────────────────────┐
 │ Vertical gyro ωz      │          │ Satellite position/vel.  │
 │ Accelerometers fx,y,z │          │ Pseudorange + Doppler    │
 │ OBD-II speed vod      │          │ Elevation + uncertainty  │
 └───────────┬───────────┘          └────────────┬─────────────┘
             │ synchronize/downsample                         │
             ▼                                                ▼
 ┌───────────────────────────┐         ┌────────────────────────────┐
 │ 3D RISS mechanization     │         │ SPP state estimation       │
 │ attitude → ENU velocity   │         │ GN-NE / GN-QR / GN-SVD     │
 │ → geodetic position       │         │ weighting + PDOP gating    │
 └─────────────┬─────────────┘         └──────────────┬─────────────┘
               │ high-rate/continuous nominal state  │ GNSS P/V fix
               └──────────────────┬───────────────────┘
                                  ▼
                 ┌────────────────────────────────┐
                 │ 9-state error-state KF         │
                 │ GM odometer SF + gyro drift    │
                 │ GNSS correction / outage DR    │
                 └────────────────────────────────┘
```

The navigation convention is local **East–North–Up (ENU)** velocity with geodetic latitude $\varphi$, longitude $\lambda$, and ellipsoidal height $h$. Angles used by the mechanization are in radians. The synchronized project products run at $\Delta t=1\ \mathrm{s}$.

---

## Module 1 — 3D RISS Mechanization in LLF

### Reduced sensor configuration

A conventional six-axis IMU integrates three gyroscopes and three accelerometers. The 3D RISS instead uses:

- one **vertical gyroscope** for azimuth rate;
- three orthogonal **accelerometers**;
- one **vehicle odometer** for forward speed $v_{od}$.

Pitch and roll are inferred from specific force rather than integrated from two horizontal gyroscopes. Odometer speed replaces acceleration integration for translational velocity. Removing those integrations reduces the position-error growth associated with a horizontal gyro bias from cubic to quadratic:

$$
\delta\theta \simeq b_\omega t,\qquad
\delta v \simeq \frac{1}{2}g b_\omega t^2,\qquad
\delta p \simeq \frac{1}{6}g b_\omega t^3,
$$

whereas direct accelerometer attitude determination removes the first integration and yields a dominant $t^2$-type position error. The architecture therefore lowers cost, size, weight, and power (SWaP), at the expense of stronger sensitivity to accelerometer and odometer noise.

### Earth model

For WGS-84 semi-major axis $a$ and eccentricity $e$, the prime-vertical and meridian radii are

$$
R_N=\frac{a}{\sqrt{1-e^2\sin^2\varphi}},\qquad
R_M=\frac{a(1-e^2)}{\left(1-e^2\sin^2\varphi\right)^{3/2}}.
$$

The implementation uses $a=6\,378\,137\ \mathrm{m}$, $e=0.08181919$, and Earth rotation rate $\omega_e=7.292115\times10^{-5}\ \mathrm{rad/s}$.

### Attitude mechanization

With the repository's body-axis/sign convention, roll $r$ and pitch $p$ are calculated as

$$
r_k=\operatorname{atan2}\!\left[-\left(f_x+v_{od}\omega_z\right),f_z\right],
$$

$$
p_k=\operatorname{atan2}\!\left[-\left(f_y-a_{od}\right),
\sqrt{f_x^2+f_z^2}\right],
\qquad
a_{od}\approx\frac{v_{od,k}-v_{od,k-1}}{\Delta t}.
$$

The azimuth/yaw $A$ is propagated from the compensated vertical gyro rate:

$$
\dot A=
\cos p\cos r\,\omega_z
-\omega_e\sin\varphi
-\frac{v^E\tan\varphi}{R_N+h}.
$$

The final terms remove the local projection of Earth rotation and the east-velocity transport rate. `dr_utils.py` applies a forward-Euler update and wraps yaw to $[0,2\pi)$.

### Velocity and geodetic position

The odometer's forward speed is resolved into LLF components:

$$
\begin{bmatrix}v^E\\v^N\\v^U\end{bmatrix}
=
\begin{bmatrix}
-\sin A\cos p\\
 \cos A\cos p\\
 \sin p
\end{bmatrix}v_{od}.
$$

The corresponding geodetic rates are

$$
\begin{bmatrix}\dot\varphi\\\dot\lambda\\\dot h\end{bmatrix}
=
\begin{bmatrix}
0 & (R_M+h)^{-1} & 0\\
[(R_N+h)\cos\varphi]^{-1} & 0 & 0\\
0&0&1
\end{bmatrix}
\begin{bmatrix}v^E\\v^N\\v^U\end{bmatrix}.
$$

At the synchronized 1 Hz rate, the code adds these rates directly at each epoch (equivalent to multiplying by $\Delta t=1\ \mathrm{s}$).

### Bias initialization and stationary detection

During a stationary interval, gravity projected into the body frame gives static accelerometer bias estimates:

$$
b_x=f_x-g\cos p\sin r,\qquad
b_y=f_y+g\sin p,\qquad
b_z=f_z+g\cos p\cos r,
$$

and the vertical gyro bias is estimated by removing the stationary Earth-rate projection,

$$
b_{\omega_z}=\omega_z-\omega_e\sin\varphi.
$$

The synchronization notebook estimates and stores these terms in `sensor_data_u.mat`. The mechanization notebook also implements a zero-speed heuristic: when `v_od == 0`, the position and attitude states are frozen and LLF velocity is set to zero. This prevents low-cost sensor noise from being integrated while parked. The report notes a nuanced result: the heuristic strongly benefits noisy MEMS data but can perturb a high-end IMU's stable, time-correlated error behavior.

### Why altitude diverges

Numerical differentiation has frequency response

$$
A(j\omega)=j\omega V(j\omega),
$$

so high-frequency odometer noise is amplified. That noise enters pitch, then vertical velocity $v^U=v_{od}\sin p$, and is finally integrated into altitude. Small pitch errors can therefore create severe vertical drift.

The repository evaluates three causal low-pass filters before differentiation:

- **moving average (MA):** $y_k=N^{-1}\sum_{i=0}^{N-1}x_{k-i}$;
- **exponential moving average (EMA):** $y_k=\alpha x_k+(1-\alpha)y_{k-1}$;
- **Butterworth IIR:** $y_k=\sum b_i x_{k-i}-\sum_{i=1}^{N}a_i y_{k-i}$.

They are paired with the implemented Tustin-style derivative

$$
\dot x_k\approx\frac{2}{\Delta t}\frac{x_k-x_{k-1}}{1+\beta}.
$$

The report's selected EMA configuration uses **$\alpha=0.4$, $\beta=0.1$**. The main notebook also exposes a second-order Butterworth candidate (`cutoff=0.2 Hz`, `fs=1 Hz`) and helper functions for MA, EMA, Butterworth, and Butterworth-plus-Tustin comparisons.

---

## Module 2 — Single-Point GPS Positioning

### Eight-state ECEF model

At each epoch, the receiver state is

$$
\mathbf{s}=
\begin{bmatrix}
x&y&z&c\delta t_r&v_x&v_y&v_z&c\delta\dot t_r
\end{bmatrix}^{T},
$$

containing ECEF position, receiver clock bias in metres, ECEF velocity, and clock drift in metres per second.

For satellite $m$, the corrected code pseudorange and Doppler-derived range rate are modeled as

$$
\rho_c^m=\|\mathbf{x}-\mathbf{x}^m\|+c\delta t_r+\varepsilon_\rho^m,
$$

$$
\dot\rho^m=\mathbf{l}^{mT}(\mathbf{v}-\mathbf{v}^m)
+c\delta\dot t_r+\varepsilon_{\dot\rho}^m,
\qquad
\dot\rho^m=-\frac{D^m c}{L_1}.
$$

The loader applies the satellite-clock correction $\rho_c=\rho+c\delta t_{sv}$, with $c=299\,792\,458\ \mathrm{m/s}$ and $L_1=1575.42\ \mathrm{MHz}$.

Stacking $M$ code and $M$ range-rate residuals produces

$$
\delta\mathbf{z}_{2M\times1}
=\mathbf{G}_{2M\times8}\delta\mathbf{s}_{8\times1}+\boldsymbol\varepsilon,
$$

with the block geometry matrix

$$
\mathbf{G}=
\begin{bmatrix}
\mathbf{L} & \mathbf{1} & \mathbf{0} & \mathbf{0}\\
\mathbf{0} & \mathbf{0} & \mathbf{L} & \mathbf{1}
\end{bmatrix},
$$

where each row of $\mathbf L$ is a receiver-to-satellite line-of-sight unit vector.

### Iterative Gauss–Newton NLS

Each epoch starts from the preceding estimate and iterates until the update norm is below tolerance or the iteration cap is reached. The checked-in notebook uses 10 iterations and $10^{-9}$ tolerance over 2,503 epochs.

Three numerical solutions are studied:

1. **Normal equations (GN-NE)**
   $$
   \delta\mathbf{s}=(\mathbf G^T\mathbf W\mathbf G)^{-1}
   \mathbf G^T\mathbf W\delta\mathbf z.
   $$
   This is inexpensive, but $\kappa(\mathbf G^T\mathbf G)=\kappa(\mathbf G)^2$; weak geometry can therefore turn a small singular value into a catastrophic error spike.

2. **QR (GN-QR)**
   $$
   \mathbf W^{1/2}\mathbf G=\mathbf Q\mathbf R,\qquad
   \mathbf R\delta\mathbf s=\mathbf Q^T\mathbf W^{1/2}\delta\mathbf z.
   $$
   QR avoids explicit formation of $\mathbf G^T\mathbf G$, but offers no truncation of near-null directions.

3. **Truncated SVD (GN-SVD / WGN-SVD)**
   $$
   \mathbf W^{1/2}\mathbf G=\mathbf U\boldsymbol\Sigma\mathbf V^T,
   \qquad
   \delta\mathbf s=\mathbf V\boldsymbol\Sigma^+
   \mathbf U^T\mathbf W^{1/2}\delta\mathbf z.
   $$
   Singular values below the implementation threshold (**0.05**) are set to zero. This regularizes nearly unobservable directions rather than amplifying them.

The active checked-in path in `nls_helper_functions.py` is **weighted truncated SVD**; normal-equation, weighted-normal-equation, and weighted-QR alternatives are retained as selectable/commented solver lines.

### Elevation weighting

Low-elevation signals traverse more atmosphere and are more vulnerable to multipath. Position and velocity observations are therefore weighted by

$$
w_i=\sin^2\theta_i,
$$

where $\theta_i$ is satellite elevation. The current implementation applies the same elevation-derived diagonal block to pseudorange and Doppler residuals.

### DOP and epoch rejection

Position DOP is obtained from the upper geometry block:

$$
\mathbf D=(\mathbf G_p^T\mathbf G_p)^+,
\qquad
PDOP=\sqrt{D_{xx}+D_{yy}+D_{zz}}.
$$

The Kalman implementation rejects a correction when **PDOP > 15** and propagates the previous nominal/error state instead. The NLS driver has the same fallback mechanism for failed epochs (fewer than four satellites), while the report also evaluates explicit PDOP-gated NLS variants. Gating limits damage from collinear/clustered satellite geometry but discards information and can lag a moving receiver.

### Recursive Kalman baseline

Project 2 also implements an eight-state constant-velocity filter:

$$
\mathbf F=\begin{bmatrix}\mathbf I_4&\mathbf I_4\\\mathbf0&\mathbf I_4\end{bmatrix},
\qquad
\hat{\mathbf x}_k^-=\mathbf F\hat{\mathbf x}_{k-1}^+,
\qquad
\mathbf P_k^-=\mathbf F\mathbf P_{k-1}^+\mathbf F^T+\mathbf Q.
$$

Pseudorange/range-rate residuals form the measurement update. Covariance symmetry and positive semidefiniteness are enforced by eigenvalue flooring. This recursive baseline smooths isolated weak-geometry epochs more effectively than unregularized GN-NE, but WGN-SVD ultimately provides the strongest NLS result.

---

## Module 3 — Loosely Coupled GNSS/3D RISS Fusion

### Error-state definition

The nominal state is propagated by 3D RISS. A linearized Kalman filter estimates

$$
\delta\mathbf x=
\begin{bmatrix}
\delta\varphi&\delta\lambda&\delta h&
\delta v^E&\delta v^N&\delta v^U&
\delta A&SF^{od}&\delta\omega_z
\end{bmatrix}^{T}.
$$

The final states represent azimuth error, odometer scale-factor error, and vertical-gyro stochastic drift in addition to position and velocity errors. Corrected odometer speed and gyro rate are

$$
\tilde v_{od}=(1+SF^{od})v_{od},\qquad
\tilde\omega_z=\omega_z-\delta\omega_z.
$$

### First-order Gauss–Markov processes

Odometer scale factor and gyro drift are modeled with exponentially decaying autocorrelation:

$$
R_{xx}(\tau)=\sigma^2e^{-\beta|\tau|},\qquad
\dot x=-\beta x+\sqrt{2\beta\sigma^2}\,w(t),
$$

or, under the first-order discrete approximation,

$$
x_k=(1-\beta\Delta t)x_{k-1}
+\sqrt{2\beta\sigma^2}\,w_k\Delta t.
$$

The correlation time is $T_c=1/\beta$, estimated from stationary autocorrelation at its $1/e$ crossing. The notebook uses:

| Process | Correlation time | Standard deviation |
|---|---:|---:|
| KVH vertical gyro | 3 h | $0.01^\circ$ (converted to rad) |
| VTI vertical gyro | 4 h | $0.1^\circ$ (converted to rad) |
| Odometer scale factor/acceleration error | 0.1 h (360 s) | 10 |

These values populate the noise-coupling matrix $\mathbf G$, including
$G_{8,8}=\sqrt{2\gamma_{od}\sigma_{od}^2}$ and
$G_{9,9}=\sqrt{2\beta_z\sigma_z^2}$.

### Linearized state transition

The transition is assembled in 3-by-3 blocks:

$$
\boldsymbol\Phi_k=
\begin{bmatrix}
\mathbf I&\mathbf F_1&\mathbf0\\
\mathbf0&\mathbf I&\mathbf F_2\\
\mathbf0&\mathbf0&\mathbf F_3
\end{bmatrix}.
$$

The implementation uses

$$
\mathbf F_1=
\begin{bmatrix}
0&\frac{\Delta t}{R_M+h}&0\\
\frac{\Delta t}{(R_N+h)\cos\varphi}&0&0\\
0&0&\Delta t
\end{bmatrix},
$$

$$
\mathbf F_2=
\begin{bmatrix}
v_{od}\cos A\cos p&-v_{od}\sin A\cos p&0\\
v_{od}\sin A\cos p& v_{od}\cos A\cos p&0\\
0&v_{od}\sin p&0
\end{bmatrix},
$$

$$
\mathbf F_3=
\begin{bmatrix}
1&0&-\Delta t\\
0&1-\gamma_{od}\Delta t&0\\
0&0&1-\beta_z\Delta t
\end{bmatrix}.
$$

### Prediction and correction

At every 1 s epoch:

1. propagate the nominal navigation solution with `riss_mechanization_3d`;
2. construct $\boldsymbol\Phi_k$ from the current geometry, odometer acceleration, yaw, and pitch;
3. predict
   $$
   \delta\hat{\mathbf x}_k^-=\boldsymbol\Phi_k\delta\hat{\mathbf x}_{k-1}^+,
   \qquad
   \mathbf P_k^-=\boldsymbol\Phi_k\mathbf P_{k-1}^+\boldsymbol\Phi_k^T
   +\mathbf G\mathbf Q\mathbf G^T;
   $$
4. form the six-component GNSS-minus-RISS position/velocity residual;
5. update using
   $$
   \mathbf K_k=\mathbf P_k^-\mathbf H^T
   (\mathbf H\mathbf P_k^-\mathbf H^T+\mathbf R_k)^{-1},
   $$
   $$
   \delta\hat{\mathbf x}_k^+=\delta\hat{\mathbf x}_k^-
   +\mathbf K_k(\mathbf z_k-\mathbf H\delta\hat{\mathbf x}_k^-),
   \qquad
   \mathbf P_k^+=(\mathbf I-\mathbf K_k\mathbf H)\mathbf P_k^-;
   $$
6. inject the error estimate into the nominal state.

The measurement matrix is $\mathbf H=[\mathbf I_{6\times6}\ \mathbf0_{6\times3}]$. GNSS-reported latitude/longitude/altitude standard deviations are converted to radians/metres to build epoch-dependent $\mathbf R_k$; the checked-in implementation empirically scales this matrix by `0.00045`.

### GNSS outage handling

When `outage_mask[k] == 1`, the code skips the GNSS-corrected solution, stores the predicted covariance/error state, and advances using RISS dead reckoning. When updates resume, the accumulated position/velocity residual drives rapid reconvergence.

The report evaluates four outages:

| Interval (s) | Duration | Behavior supported by the report |
|---:|---:|---|
| 100–280 | 180 s (3 min) | Extended outage; horizontal error enters the reported $>300$–$500$ m regime |
| 1,000–1,150 | 150 s (2.5 min) | Drift remains within a few hundred metres |
| 1,500–1,590 | 90 s (1.5 min) | Shortest outage; notable but bounded drift |
| 2,000–2,120 | 120 s (2 min) | Drift remains within a few hundred metres |

The report does not provide a separate endpoint error or fitted drift rate for each interval; those quantities should be computed from the generated outage error arrays rather than inferred from the figures. Its defensible aggregate conclusion is that error growth is nonlinear and accelerates with outage duration: 90–150 s outages stay within a few hundred metres, while the 180 s outage exceeds roughly 300–500 m before GNSS-aided recovery.

---

## Performance Benchmarks and Key Findings

All errors below are relative to the provided NovAtel integrated/reference solution. Position error conversion uses

$$
\delta N=\delta\varphi(R_M+h),\qquad
\delta E=\delta\lambda(R_N+h)\cos\varphi,
$$

with $e_{2D}=\sqrt{\delta E^2+\delta N^2}$ and
$e_{3D}=\sqrt{e_{2D}^2+\delta h^2}$.

### Project 1 — standalone 3D RISS

The report provides the following tactical/high-end versus automotive MEMS RMS comparison. Values are metres; percentages in the report are omitted here for compactness.

| Configuration | Sensor | East RMS | North RMS | 2D RMS | Altitude RMS | 3D RMS |
|---|---|---:|---:|---:|---:|---:|
| Baseline RISS core | Tactical/high-end | 13.7 | 27.4 | 30.6 | 101.5 | 106.0 |
| Baseline RISS core | VTI MEMS | 1,379.0 | 613.4 | 1,509.4 | 314.3 | 1,541.8 |
| Bias estimation | Tactical/high-end | 69.7 | 137.9 | 154.5 | 123.8 | 198.0 |
| Bias estimation | VTI MEMS | 214.4 | 746.3 | 776.4 | 306.8 | 834.8 |
| Stationary freeze | Tactical/high-end | 118.2 | 203.0 | 235.0 | 101.5 | 256.0 |
| Stationary freeze | VTI MEMS | 746.6 | 752.9 | 1,060.4 | 314.3 | 1,105.9 |
| Bias + stationary freeze | Tactical/high-end | 97.1 | 204.1 | 226.0 | 123.8 | 258.0 |
| Bias + stationary freeze | VTI MEMS | **255.2** | **671.5** | **718.3** | **306.8** | **781.1** |

The saved baseline output in `Mechanization_F.ipynb` additionally retains tactical/high-end absolute maxima:

| Metric | East (m) | North (m) | 2D (m) | Altitude (m) | 3D (m) |
|---|---:|---:|---:|---:|---:|
| Tactical/high-end maximum | 35.05 | 52.80 | 54.81 | **159.54** | 163.83 |
| VTI MEMS maximum | Not reported | Not reported | Not reported | Not reported | Not reported |

| Velocity metric | East (m/s) | North (m/s) | Up (m/s) |
|---|---:|---:|---:|
| RMS | 0.326 | 0.310 | 0.366 |
| Maximum | 1.839 | 1.782 | 2.346 |

All filtered cases below use the report's Tustin parameter $\beta=0.1$:

| Odometer prefilter | Tactical/high-end altitude RMS (m) | VTI MEMS altitude RMS (m) |
|---|---:|---:|
| Baseline derivative | 101.5 | **306.8** |
| MA, $N=3$ | 21.1 | 396.9 |
| EMA, $\alpha=0.4$ | 20.6 | 467.3 |
| 2nd-order Butterworth, $f_c=0.2\ \mathrm{Hz}$ | **20.4** | 402.9 |

EMA + Tustin reduces tactical/high-end altitude RMS from **101.5 m to 20.6 m**, a **4.9× (about fivefold)** improvement; Butterworth + Tustin reaches a similar 20.4 m. In contrast, every tested prefilter worsens the VTI altitude RMS relative to its 306.8 m baseline. This supports the report's interpretation that the low-end stream's larger, time-varying errors are amplified by differentiation and compounded by filter lag. Static bias compensation still substantially improves the VTI horizontal/3D solution, even though the zero-speed freeze by itself does not outperform that bias-only case in the tabulated RMS results.

### Project 2 — numerical conditioning and SPP

Selected position results from the report/error log are shown below. Values are metres.

| Method | 2D RMS | 2D max | Altitude RMS | Altitude max | 3D RMS | 3D max |
|---|---:|---:|---:|---:|---:|---:|
| GN-NE (unprotected) | 19.79 | **663.87** | 51.37 | **2,428.05** | 55.05 | **2,517.17** |
| KF (unprotected) | 12.85 | 56.16 | 12.41 | 85.57 | 17.87 | 87.09 |
| DOP-gated GN-NE, PDOP 15 | 14.32 | 60.57 | 9.80 | 52.49 | 17.35 | 60.58 |
| DOP-gated KF, PDOP 15 | 13.03 | 62.52 | 11.65 | 66.43 | 17.48 | 69.59 |
| GN-SVD | 14.03 | 48.07 | 9.78 | 52.49 | 17.10 | 52.59 |
| **WGN-SVD** | **13.21** | **48.07** | **8.96** | **32.57** | **15.96** | **49.44** |

| Method | East velocity RMS / max | North velocity RMS / max | Up velocity RMS / max |
|---|---:|---:|---:|
| GN-NE | 2.50 / 124.46 | 1.34 / 65.58 | 10.18 / 508.45 |
| KF | 0.23 / 5.32 | 0.31 / 7.60 | 0.42 / 4.05 |
| GN-SVD | 0.16 / 2.43 | 0.25 / 7.72 | 0.30 / 3.56 |
| **WGN-SVD** | **0.15 / 2.43** | **0.25 / 7.72** | **0.27 / 3.56** |

Key conclusions:

- GN-NE's low average cost masks severe vulnerability to high condition numbers; its maximum 3D error exceeds 2.5 km.
- PDOP gating removes the worst epochs and dramatically lowers maxima, but throws away measurements.
- QR avoids explicitly squaring the condition number but does not regularize near-null singular directions; in this dataset it does not remove the observed spikes.
- Truncated SVD suppresses near-zero singular-value blow-up.
- Elevation-weighted SVD gives the best overall NLS result and slightly outperforms the reported DOP-gated KF in 3D RMS and maximum error.

### Project 3 — nominal fusion and outage resilience

The checked-in nominal VTI GNSS/RISS notebook output reports:

| Metric | East | North | 2D | Altitude | 3D |
|---|---:|---:|---:|---:|---:|
| Position RMS (m) | 5.12 | 5.98 | **7.87** | 1.03 | 7.94 |
| Position maximum (m) | 15.04 | 16.42 | **19.27** | 7.87 | 19.27 |

| Metric | East (m/s) | North (m/s) | Up (m/s) |
|---|---:|---:|---:|
| Velocity RMS | 0.397 | 0.403 | 0.273 |
| Velocity maximum | 2.390 | 8.562 | 3.305 |

Thus the automotive MEMS fused baseline achieves approximately **8 m horizontal RMS** and sub-1 m/s RMS velocity on all three axes. During simulated loss of GNSS, dead-reckoning drift grows to hundreds of metres and can exceed **300–500 m** for the longest outage, followed by fast correction after reacquisition.

> The saved notebook applies an additional 95% GNSS / 5% INS velocity blend after the Kalman-filter call. The tabulated saved velocity errors therefore describe that executed notebook output, not a pure unmodified KF output.

---

## Repository Structure

```text
.
├── README.md
├── project 1/
│   ├── EE523 Project I.pdf                    # assignment specification
│   ├── paper/
│   │   ├── 1_Elaraby.pdf                      # final Project 1 report
│   │   └── IEEE-conference-template-062824/
│   │       ├── IEEE-conference-template-062824.tex
│   │       └── plots/                          # RISS block diagram, setup, altitude/results
│   └── src/
│       ├── Data Synchronization.ipynb          # align, bias-estimate, average/downsample, save MAT
│       ├── Mechanization.ipynb                 # earlier/base mechanization workflow
│       ├── Mechanization_F.ipynb               # main mechanization, metrics, plots, filter switches
│       ├── pitch_trials.ipynb                  # MA/EMA/Butterworth/Tustin experiments
│       ├── dr_utils.py                         # RISS attitude, velocity, position, causal filters
│       ├── error_calc.py                       # position/velocity max, RMS, and standard deviation
│       ├── error_calc2.py                      # adds path length and percentage error
│       ├── sensor_data.mat                     # synchronized product (legacy schema)
│       └── sensor_data_u.mat                   # synchronized 2,496-epoch product with biases
├── project 2/
│   ├── EE523 Project II.pdf
│   ├── paper/conference-latex-template/
│   │   ├── 2_Elaraby.pdf                       # final Project 2 report
│   │   └── IEEE-conference-template-062824/
│   │       ├── IEEE-conference-template-062824.tex
│   │       └── plots/                          # DOP, conditioning, trajectory, P/V errors
│   └── src/
│       ├── data_loading.ipynb                  # raw MAT → satellite_data.npz
│       ├── dataset_analysis.ipynb              # satellite visibility/geometry/DOP exploration
│       ├── experimentation.ipynb               # NLS, KF, error metrics, and all result plots
│       ├── satellite_data.npz                  # 2,503-epoch raw/reference Python product
│       ├── utils.py                            # data extraction and ECEF/geodetic/ENU transforms
│       ├── solver_utils.py                     # LOS, residual, geometry, and DOP routines
│       ├── nls_helper_functions.py             # weighted GN with QR/SVD and trajectory driver
│       ├── kf_helper_functions.py              # 8-state constant-velocity KF
│       ├── error_calc.py                       # geodetic P/V error statistics
│       └── Error Logs.txt                      # retained solver/gating benchmark results
├── project 3/
│   ├── EE523 Project III.pdf
│   ├── paper/
│   │   ├── 3_Elaraby.pdf                       # final Project 3 report
│   │   └── conference-latex-template/IEEE-conference-template-062824/
│   │       ├── IEEE-conference-template-062824.tex
│   │       └── plots/                          # nominal and outage figures
│   └── src/
│       ├── Data Synchronization.ipynb          # IMU/odometer/GNSS/reference → sensor_data.mat
│       ├── experimentation.ipynb               # filter setup, execution, metrics, and plots
│       ├── kf_helper_funcs.py                  # RISS, Φ blocks, R, and residual construction
│       ├── kf_integration.py                   # 9-state predict/update and outage branching
│       ├── error_calc.py                       # fused P/V error statistics
│       ├── sensor_data.mat                     # synchronized 2,496-epoch fusion product
│       └── plots/                              # generated nominal/outage PDFs
└── Dataset/
    ├── 1. Project data Info.pdf                # synchronization/units/sign guidance
    ├── 2. KVH-1750-IMU-datasheet.pdf
    ├── 3. VTI_SCC1300-d04_IMU-datasheet.pdf
    ├── 4. Instructions on presenting the project outputs.pdf
    ├── 5. Instructions for project 2.pdf        # supplied KF tuning example
    ├── 6. Q and R Document for 3D RISS.pdf
    ├── Dataset notes.pdf                       # variable map and sampling rates
    ├── NovAtel/
    │   ├── 1.RAWIMU.mat                        # high-end IMU, 200 Hz
    │   ├── 2.BESTGPSPOS.mat / 2.BESTGPSVEL.mat # 1 Hz GNSS reference/fusion updates
    │   ├── 3.INSPVA_Reference.mat              # 5 Hz integrated ground truth
    │   ├── GPS_data_combined_Nova.mat          # SPP satellite observables and states
    │   └── *.ASC, ephemeris, SATVIS, SATXYZ    # source logs and auxiliary products
    ├── OBDII_data/
    │   ├── CarChip_Speed.mat                   # 1 Hz forward speed, m/s
    │   └── odo_second.mat                      # synchronization timestamps
    └── TPI/
        ├── TPI_data_interpolated2.mat          # VTI IMU, 20 Hz
        ├── 1.TPI_data_denoised_LOD6_interpolated2.mat
        ├── 2.TPI_GPS_soln.mat
        ├── 3.TPI_processed_soln.mat
        ├── TPI_bar_data.mat / TPI_mag_data.mat # auxiliary barometer/magnetometer data
        └── 5. TPI and reference.ipynb          # dataset field inspection
```

`project 4/` is outside the scope of this README.

---

## Setup and How to Run

### 1. Create a Python environment

The notebooks were originally executed with Python 3.10, NumPy 1.23.3, and SciPy 1.9.1. The implementation also uses Matplotlib and Jupyter.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install the confirmed dependencies:

```bash
python -m pip install numpy scipy matplotlib jupyter ipykernel
```

Run notebooks **from their own `src/` directory**. Their imports and data paths are relative to that location, and directory names contain spaces.

### 2. Reproduce Project 1 — mechanization

```bash
cd "project 1/src"
jupyter notebook "Data Synchronization.ipynb"
```

Run all cells to:

1. load KVH/NovAtel, VTI, odometer, and INSPVA reference data;
2. correct the documented KVH axis sign;
3. find the common timestamp interval;
4. average/downsample 200 Hz KVH, 20 Hz VTI, and 5 Hz reference data to the 1 Hz odometer grid;
5. estimate static biases; and
6. write `sensor_data_u.mat`.

Then launch the principal mechanization notebook:

```bash
jupyter notebook "Mechanization_F.ipynb"
```

Run all cells to reproduce latitude/longitude/altitude, ENU velocity, attitude, trajectory, and error plots. In the `dead_reckoning` cell:

- use `compute_attitude(...)` for the baseline derivative;
- select `compute_attitude_ema(..., alpha=0.4, beta=0.1)` for the report's chosen altitude mitigation;
- select `compute_attitude_bw(...)` for Butterworth + Tustin; and
- set `stop_mechanization=True` in a run call to enable zero-speed freezing.

Use `pitch_trials.ipynb` to reproduce the MA/EMA/Butterworth parameter sweeps and pitch-error/SNR comparisons.

### 3. Reproduce Project 2 — SPP and conditioning

The ready-to-run `satellite_data.npz` is already included. To rebuild it from raw MAT files:

```bash
cd "project 2/src"
jupyter notebook "data_loading.ipynb"
```

Then run:

```bash
jupyter notebook "experimentation.ipynb"
```

The notebook:

1. runs NLS for 2,503 epochs;
2. converts ECEF estimates to geodetic/ENU;
3. runs the constant-velocity Kalman filter;
4. calculates DOP, condition number, position error, and velocity error; and
5. draws the trajectory, DOP/conditioning, covariance, position, and velocity figures.

Solver selection is made in `nls_helper_functions.py` inside `nonlinear_least_squares`:

```python
# Normal equations
delta_x = np.linalg.pinv(G.T @ G) @ G.T @ residuals

# Weighted normal equations
delta_x = np.linalg.pinv(G.T @ W @ G) @ G.T @ W @ residuals

# Weighted truncated SVD (checked-in active path)
delta_x = gauss_newton_svd_weighted(G, residuals, W)

# Weighted QR
delta_x = gauss_newton_qr_weighted(G, residuals, W)
```

Enable/adjust the NLS PDOP check in the same function to reproduce gated NLS variants. The KF's active gate is `PDOP > 15` in `kf_helper_functions.py`. `dataset_analysis.ipynb` independently reproduces satellite-count and DOP analyses.

Uncomment the notebook `plt.savefig(...)` lines to regenerate the PDFs under the report's `plots/` directory.

### 4. Reproduce Project 3 — GNSS/RISS integration

To rebuild the synchronized fusion input:

```bash
cd "project 3/src"
jupyter notebook "Data Synchronization.ipynb"
```

This creates `sensor_data.mat`, containing 2,496 one-second epochs of high-/low-end IMU, odometer, GNSS, and reference states.

Run the fusion:

```bash
jupyter notebook "experimentation.ipynb"
```

The checked-in notebook is configured for VTI denoised/interpolated IMU data and nominal GNSS availability. To reproduce the four report outages, change the mask cell to:

```python
outage_mask = np.zeros(num_epochs, dtype=int)
outage_mask[100:280] = 1
outage_mask[1000:1150] = 1
outage_mask[1500:1590] = 1
outage_mask[2000:2120] = 1
```

Run all subsequent cells to regenerate nominal/outage trajectory, component-error, 2D/3D-error, velocity-error, and covariance plots. For a clean comparison of the Kalman filter itself, disable the notebook's post-filter 95/5 GNSS/INS velocity blend before recomputing velocity statistics.

### 5. Plot conventions

For publication-compatible figures:

- plot elapsed time from trajectory start, preferably in minutes;
- convert latitude/longitude errors to north/east metres;
- report standard deviation, RMS, and absolute maximum;
- show 2D and 3D magnitudes separately; and
- overlay outage intervals when presenting dead-reckoning drift.

---

## Implementation Notes and Reproducibility Caveats

- **Sampling:** raw KVH/NovAtel IMU is 200 Hz, VTI is 20 Hz, INSPVA reference is 5 Hz, and odometer/GNSS updates are 1 Hz. Synchronization uses the latest stream start, earliest stream end, and block averaging to 1 Hz.
- **Units:** accelerometers are in $\mathrm{m/s^2}$, gyroscopes in $\mathrm{rad/s}$, speed in $\mathrm{m/s}$, geodetic computations in radians, and reference attitude fields originate in degrees.
- **Axis correction:** dataset documentation requires inversion of KVH `f.y` and `w.y`; the synchronization code applies the relevant sign convention to used channels.
- **Notebook state matters:** some report variants are selected by commenting/uncommenting solver or filter calls rather than through a central configuration file.
- **Project 3 outage mask:** the checked-in notebook writes zeros into all four candidate slices. Because `kf_integration.py` interprets `1` as outage, set those slices to one to activate the published outage test.
- **Project 3 covariance implementation:** `kalman_predict` manually zeros rows 6–8 of the predicted covariance, and the correction scales the same rows. This is an experimental implementation detail, not a generic error-state KF requirement.
- **Project 3 saved velocity metrics:** a post-KF velocity blend is applied in the notebook. Position metrics are from the fused state; saved velocity metrics include that blend.
- **No fabricated outage rates:** the report gives exact outage durations and aggregate error ranges, but not a per-window endpoint table. Compute any desired $\mathrm{m/s}$ drift slopes directly from regenerated `position_errors['2d']`.
- **Generated artifacts:** paper PDFs and plots are retained for reference; source notebooks and Python modules are the reproducible implementation.

---

## Main Findings

1. **Reduced inertial sensing is viable for land vehicles.** Eliminating two horizontal gyroscopes and using odometer speed reduces SWaP and removes integration stages that otherwise produce cubic gyro-bias position growth.
2. **Vertical navigation is the weak mode of unaided 3D RISS.** Differentiating 1 Hz odometer speed amplifies noise; causal smoothing plus a Tustin-style derivative reduces tactical altitude RMS from 101.5 m to 20.6 m, but low-end MEMS still requires stronger modeling/aiding.
3. **GNSS geometry is also a numerical-conditioning problem.** Normal equations square the Jacobian condition number and create extreme spikes under poor satellite geometry.
4. **Weighted truncated SVD is the most robust SPP solver evaluated.** It retains usable measurement information, rejects near-null singular directions, and downweights low-elevation satellites.
5. **GNSS/RISS fusion bounds long-term drift.** The VTI nominal fused solution reaches about 7.9 m horizontal RMS, but 90–180 s GNSS outages expose rapid dead-reckoning degradation into the hundreds of metres. Reacquired GNSS measurements quickly pull the filter back toward the reference trajectory.

---

## Reports

The full derivations, figures, and discussion are available in:

- [`project 1/paper/1_Elaraby.pdf`](project%201/paper/1_Elaraby.pdf)
- [`project 2/paper/conference-latex-template/2_Elaraby.pdf`](project%202/paper/conference-latex-template/2_Elaraby.pdf)
- [`project 3/paper/3_Elaraby.pdf`](project%203/paper/3_Elaraby.pdf)
