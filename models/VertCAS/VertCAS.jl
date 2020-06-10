# # Vertical Collision Avoidance System
#
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`VertCAS.ipynb`](__NBVIEWER_ROOT_URL__/../VertCAS.ipynb)
#
#
# This example [1, 2] considers commercial aircraft, which are required to operate
# with a collision avoidance system that gives vertical climbrate advisories to
# pilots to prevent near midair collisions (NMACs). An NMAC occurs when the
# aircraft are separated by less than 100 ft vertically and 500 ft horizontally.

# ## Model
#
# The model considers two aircraft: an ownship aircraft equipped with
# VerticalCAS, and an intruder aircraft. In this formulation, the intruder is
# assumed to maintain level flight. The system uses four variables to describe
# the encounter with the intruder aircraft:
#
# 1)``h(ft)``: Intruder altitude relative to ownship``[−3000,3000]``
# 2)``\dot h\0(ft/min)``: Ownship vertical climbrate``[−2500,2500]``
# 3)``τ(s)``: Time to loss of horizontal separation``[0,40]``
# 4)``s\_{adv}``: Previous advisory from VerticalCAS
# The first two state variables describe the encounter geometry vertically.
# The ``τ`` variable condenses the horizontal geometry into a single variable
# by providing a countdown until the intruder will no longer be separated
# horizontally, at whichpoint the ownship must be vertically separated to avoid
# an NMAC. 
# The ``s\_{adv}`` variable is categorical and can be any one of the nine
# possible advisories given by the system, and conditioning the next advisory
# on the current advisory allows the system to maintain consistency when
# alerting pilots. The nine possiblea dvisories are:
#
# 1) COC: Clear of Conflic
# 2) DNC: Do Not Climb
# 3) DND: Do Not Descent
# 4) DES1500: Descend at least 1500 ft/min
# 5) CL1500: Climb at least 1500 ft/min
# 6) SDES1500: Strengthen Descent to at least 1500 ft/min
# 7) SCL1500: Strengthen Climb to at least 1500 ft/min
# 8) SDES2500: Strengthen Descent to at least 2500 ft/min
# 9) SCL2500: Strengthen Climb to at least 2500 ft/min
#
# Each advisory instructs the pilot to accelerate until comply-ing with the
# specified climb or descent rate, except for COC, which allows the pilot
# freedom to choose any acceleration ``\ddot h\_0 ∈ [−g/8, g/8]``, where ``g``
# is the sea-level gravitational acceleration constant.
# For advisories DNC, DND, DES1500, and CL1500, the pilot is assumed to
# accelerate in the range ``|a| ∈ [g/4,g/3]`` with the sign of ``\ddot h\_0``
# determined by the specific advisory. If the pilot is already compliant with
# the given advisory, then the pilot is assumed to continue at the current
# climbrate. For advisories SDES1500, SCL1500, SDES2500, and SCL2500, the pilot
# as assumed to accelerate at ``±g/3`` until compliance. For example, a pilot
# receiving the CL1500 advisory while descending at −500 ft/min is assumed to
# begin accelerating upwards with some acceleration between ``g/4`` and ``g/3``
# and then maintaining a constant climbrate upon reaching the 1500 ft/min
# climbrate. New advisories ``s\_{adv}`` are given once each second ``(∆t= 1)``
#

using NeuralNetworkAnalysis

@taylorize function VCAS!(dx, x, p, t)
    x₁, x₂, x₃ = x
    dx[1] = x₁
    dx[2] = x₂
    dx[3] = x₃
end

# define the initial-value problem
##X₀ = Hyperrectangle(low=[...], high=[...])

##prob = @ivp(x' = VCAS!(x), dim: ?, x(0) ∈ X₀)

# solve it
##sol = solve(prob, T=0.1);

# ## Specifications
#
# The verification objective of this system is that the ownship prevents an NMAC

# ## Results

# ## References

# [1] [Julian, K. D., & Kochenderfer, M. J. (2019). A reachability method for
# verifying dynamical systems with deep neural network controllers.
# arXiv preprint arXiv:1903.00520.](https://arxiv.org/pdf/1903.00520.pdf).
# [2] Akintunde, M. E., Botoeva, E., Kouvaros, P., & Lomuscio, A. (2020, May).
# [Formal Verification of Neural Agents in Non-deterministic Environments.
# In Proceedings of the 19th International Conference on Autonomous Agents and
# MultiAgent Systems (pp. 25-33).](http://ifaamas.org/Proceedings/aamas2020/pdfs/p25.pdf)
#
