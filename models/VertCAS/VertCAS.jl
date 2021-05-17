# # Vertical Collision Avoidance (VCAS)
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/VertCAS.ipynb)
#
# This example [1, 2] considers commercial aircraft, which are required to operate
# with a collision avoidance system that gives vertical climbrate advisories to
# pilots to prevent near midair collisions (NMACs). An NMAC occurs when the
# aircraft are separated by less than 100 ft vertically and 500 ft horizontally.

# ## Model
#
# This benchmark is a closed-loop variant of aircraft collision avoidance
# system ACAS X. The scenario involves two aircraft, the ownship and the
# intruder, where the ownship is equipped with a collision avoidance system
# referred to as VerticalCAS [3]. VerticalCAS once every second issues vertical
# climbrate advisories to the ownship pilot to avoid a near mid-air collision
# (NMAC), a region where the ownship and intruder are separated by less than
# 100ft vertically and 500ft horizontally. The ownship (black) is assumed
# to have a constant horizontal speed, and the intruder (red) is assumed to
# follow a constant horizontal trajectory towards ownship, see Figure 1.
# The current geometry of the system is described by
#
# 1) ``h(ft)``: Intruder altitude relative to ownship
# 2) ``\dot h_0 (ft/min)``: Ownship vertical climbrate
# 3) ``τ(s)``: the seconds until the ownship (black) and intruder (red) are no longer horizontally separated
#
# We can, therefore, assume that the intruder is static and the horizontal
# separation ``\tau`` decreases by one each second.
# There are 9 advisories and each of them instructs the pilot to accelerate
# until the vertical climbrate of the ownship complies with the advisory:
#
# 1) COC: Clear of Conflict
# 2) DNC: Do Not Climb
# 3) DND: Do Not Descend
# 4) DES1500: Descend at least 1500 ft/min
# 5) CL1500: Climb at least 1500 ft/min
# 6) SDES1500: Strengthen Descent to at least 1500 ft/min
# 7) SCL1500: Strengthen Climb to at least 1500 ft/min
# 8) SDES2500: Strengthen Descent to at least 2500 ft/min
# 9) SCL2500: Strengthen Climb to at least 2500 ft/min
#
# In addition to the parameters describing the geometry of the encounter, the
# current state of the system stores the advisory ``adv`` issued to the ownship
# at the previous time step. VerticalCAS is implemented as nine ReLU networks
# ``N_i``, one for each (previous) advisory, with three inputs
# ``(h,\dot{h}_0,\tau)``, five fully-connected hidden layers of 20 units each,
# and nine outputs representing the score of each possible advisory. Therefore,
# given a current state ``(h,\dot{h}_0,\tau,\text{adv})``, the new advisory
# ``adv`` is obtained by computing the argmax of the output of ``N_{\text{adv}}``
# on ``(h,\dot{h}_0,\tau)``.
# Given the new advisory, if the current climbrate does not comply with it, the
# pilot can choose acceleration ``\ddot{h}_0`` from the given set:
#
# 1) COC: ``\{-\frac{g}{8}, 0, \frac{g}{8}\}``
# 2) DNC: ``\{-\frac{g}{3}, -\frac{7g}{24}, -\frac{g}{4}\}``
# 3) DND: ``\{\frac{g}{4}, \frac{7g}{24}, \frac{g}{3}\}``
# 4) DES1500: ``\{-\frac{g}{3}, -\frac{7g}{24}, -\frac{g}{4}\}``
# 5) CL1500: ``\{\frac{g}{4}, \frac{7g}{24}, \frac{g}{3}\}``
# 6) SDES1500: ``\{-\frac{g}{3}\}``
# 7) SCL1500: ``\{\frac{g}{3}\}``
# 8) SDES2500: ``\{-\frac{g}{3}\}``
# 9) SCL2500: ``\{\frac{g}{3}\}``
#
# where $g$ represents the gravitational constant ``32.2 \ \text{ft/s}^2``.
# If the new advisory is COC(1), then it can be any acceleration from the set ``{−g/8, 0, g/8}``.
# For all remaining advisories, if the previous advisory coincides with the new one and the
# current climb rate complies with the new advisory (e.g., ``\dot{h}_0`` is non-positive for DNC and
# ``\dot{h}_0 ≥ 1500`` for CL1500) the acceleration `\ddot{h}_0`` is ``0``.
#
# Given the current system state ``(h,\dot{h}_0,\tau,\text{adv})``, the new
# advisory ``\text{adv}'`` and the acceleration ``\ddot{h}_0``, the new state
# of the system ``(h(t+1),\dot{h}_0(t+1),\tau(t+1),\text{adv}(t+1))`` can
# be computed as follows:
#
# ```math
# \begin{aligned}
# h(t+1) &=& h - \dot{h}_0 \Delta\tau - 0.5 \ddot{h}_0 \Delta\tau^2 \\
# \dot{h}_0(t+1) &=& \dot{h}_0 + \ddot{h}_0 \Delta\tau \\
# \tau(t+1) &=& \tau - \Delta\tau \\
# \text{adv}(t+1) &=& \text{adv}'
# \end{aligned}
# ```
# where ``\Delta\tau=1``.
#

using NeuralNetworkAnalysis

@taylorize function VCAS!(dx, x, p, t)
    tau, adv = x
    dx[1] = x₁
    dx[2] = x₂
    dx[3] = x₃
end

# ## Specifications
#
# For this benchmark the aim is to verify that the ownship avoids entering the
# NMAC zone after ``k \in \{1, \dots, 10\}`` time steps, i.e., ``h(k) > 100`` or
# ``h(k) < -100``, for all possible choices of acceleration by the pilot. The
# set of initial states considered is as follows: ``h(0) \in [-133, -129]``,
# ``\dot{h}_0(0) \in \{-19.5, -22.5, -25.5, -28.5\}``, ``\tau(0) = 25`` and
# ``\text{adv}(0) = \text{COC}``.

# ## Results

# define the initial-value problem
##X₀ = Hyperrectangle(low=[...], high=[...])

##prob = @ivp(x' = VCAS!(x), dim: ?, x(0) ∈ X₀)

# solve it
##sol = solve(prob, T=0.1);

# ## References

# [1] [Julian, K. D., & Kochenderfer, M. J. (2019). A reachability method for
# verifying dynamical systems with deep neural network controllers.
# arXiv preprint arXiv:1903.00520.](https://arxiv.org/pdf/1903.00520.pdf)
#
# [2] Akintunde, M. E., Botoeva, E., Kouvaros, P., & Lomuscio, A. (2020, May).
# [Formal Verification of Neural Agents in Non-deterministic Environments.
# In Proceedings of the 19th International Conference on Autonomous Agents and
# MultiAgent Systems (pp. 25-33).](http://ifaamas.org/Proceedings/aamas2020/pdfs/p25.pdf).
#
