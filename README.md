# Risk-Aware-Safety-Critical-Control-for-AVs-via-Wasserstein-Barycenter (WB-CVaR-CBF)
Paper:Risk-Aware Safety-Critical Control for Autonomous Vehicles via Wasserstein Barycenter with Experimental and Numerical Analysis

## Overview
This repository accompanies the paper proposing an MPC-based safety filter that fuses multi-sensor/cooperative perception with a 2-Wasserstein barycenter and enforces safety using a CVaR-regularized Control Barrier Function (CBF). The result is a risk-aware, tractable optimization that preserves safety under localization noise and biased V2X measurements, validated in on-vehicle trials and Monte Carlo simulations.

## Key ideas

- Cooperative perception with Wasserstein barycenter
Sensor measurements (LiDAR/Camera/V2X) are modeled as Gaussians, including potential V2X bias from latency and network effects. Their unified obstacle distribution is computed via the p-Wasserstein metric (p=2) and barycenter.

- Risk-aware safety via CVaR–CBF
The CBF safety constraint—stochastic due to measurement uncertainty—is posed as a chance constraint; CVaR provides a convex, tractable surrogate to penalize tail events (worst-case safety violations).

- MPC + online safety filter
An MPC node produces a nominal command; a safety-filter QP minimally modifies it to satisfy CVaR–CBF constraints each control cycle (ROS 2 architecture).

## Testing video

![CVaR_nice-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/3734ea06-f6bf-47a3-84c3-9a1923af6257)

