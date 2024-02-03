# Full, non-simplified dynamics:
@taylorize function Quadrotor_full!(dx, x, p, t)
    x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀, x₁₁, x₁₂, u₁, u₂, u₃ = x

    F₁ = g + u₁ / m
    Tx = u₂ / Jx
    Ty = u₃ / Jy
    sx7 = sin(x₇)
    cx7 = cos(x₇)
    sx8 = sin(x₈)
    cx8 = cos(x₈)
    sx9 = sin(x₉)
    cx9 = cos(x₉)
    sx7sx9 = sx7 * sx9
    sx7cx9 = sx7 * cx9
    cx7sx9 = cx7 * sx9
    cx7cx9 = cx7 * cx9
    sx7cx8 = sx7 * cx8
    cx7cx8 = cx7 * cx8
    sx7_cx8 = sx7 / cx8
    cx7_cx8 = cx7 / cx8
    x4cx8 = cx8 * x₄
    p11 = sx7_cx8 * x₁₁
    p12 = cx7_cx8 * x₁₂
    xdot9 = p11 + p12

    dx[1] = (cx9 * x4cx8 + (sx7cx9 * sx8 - cx7sx9) * x₅) + (cx7cx9 * sx8 + sx7sx9) * x₆
    dx[2] = (sx9 * x4cx8 + (sx7sx9 * sx8 + cx7cx9) * x₅) + (cx7sx9 * sx8 - sx7cx9) * x₆
    dx[3] = (sx8 * x₄ - sx7cx8 * x₅) - cx7cx8 * x₆
    dx[4] = (x₁₂ * x₅ - x₁₁ * x₆) - g * sx8
    dx[5] = (x₁₀ * x₆ - x₁₂ * x₄) + g * sx7cx8
    dx[6] = (x₁₁ * x₄ - x₁₀ * x₅) + (g * cx7cx8 - F₁)
    dx[7] = x₁₀ + sx8 * xdot9
    dx[8] = cx7 * x₁₁ - sx7 * x₁₂
    dx[9] = xdot9
    dx[10] = Cyzx * (x₁₁ * x₁₂) + Tx
    dx[11] = Czxy * (x₁₀ * x₁₂) + Ty
    dx[12] = Cxyz * (x₁₀ * x₁₁) + Tz
    dx[13] = zero(u₁)
    dx[14] = zero(u₂)
    dx[15] = zero(u₃)
    return dx
end
