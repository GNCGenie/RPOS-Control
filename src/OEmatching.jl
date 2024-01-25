# ╔═╡ b8697459-42d5-4a79-b563-f10174a1fc53
begin
    using LinearAlgebra
    using Optim
    using SatelliteDynamics
    using GLMakie
end

# ╔═╡ b77d32b9-e6c8-4f25-864d-354441744839
begin
    ωₑ = 7.292115146706979e-5
    μ = 3.986004415e14
    Rₑ = 6.378136300e6
    J₂ = 0.0010826358191967
end

# ╔═╡ 5d02db4f-3049-4da1-9f1a-b478f99381b6
begin
    """
 Elementary functions for Orbital dynamics
 """
    Velocity(a) = √(μ / a)
    Velocity(a, r) = √(μ * (2 / r - 1 / a))
    AngularVelocity(a) = √(μ / a^3)
    TimePeriod(a) = 2π / AngularVelocity(a)
    Eccentricity(rₚ, rₐ) = (rₐ - rₚ) / (rₐ + rₚ)
    SemiMajorAxis(T) = (μ * (T / 2π)^2)^(1 / 3)
    Δv(vᵢ, vₑ) = norm(vᵢ - vₑ)
    rₚ(a, e) = a * (1 - e)
end

# ╔═╡ 7a041712-8b2b-4de7-94be-f77c9b715e78
begin
    """
 We define a structure to store Keplerian orbital elements.
 Alongside storing the 6 Classical Orbit elements, it computes specific angular momentum semi latus rectum and radial distance from origin 

 The default constructor required :
 a : Semi Major Axis
 e : Eccentricity (Default = 0)
 i : Inclination (Default = 0)
 ω : Argument of periapsis (Default = 0)
 Ω : RAAN (Default = 0)
 θ : True anomaly (Default = 0)
 """
    struct OE{T<:Real}
        a::T # Semi Major Axis
        e::T # Eccentricity
        i::T # Inclination
        ω::T # Argument of periapsis
        Ω::T # RAAN
        θ::T # True anomaly
        p::T # Semi latus rectum
        r::T # Radial distance
        h::T # Specific angular momentum
    end

    """
 Constructors & Utility Functions for doing operations on OEs
 """
    function OE(a, e=0.0, i=0.0, ω=0.0, Ω=0.0, θ=0.0)
        p = a * (1 - e^2)
        r = p / (1 + e * cos(θ))
        h = a * Velocity(a)

        OE(a, e, i, ω, Ω, θ, p, r, h)
    end
    function OE(x::Array)
        OE(x...)
    end

    function Base.:-(oeᵢ::OE, oeₜ::OE)::Array
        da = oeᵢ.a - oeₜ.a
        de = oeᵢ.e - oeₜ.e
        di = oeᵢ.i - oeₜ.i
        dω = acos(cos(oeᵢ.ω - oeₜ.ω))
        dΩ = acos(cos(oeᵢ.Ω - oeₜ.Ω))
        dθ = acos(cos(oeᵢ.θ - oeₜ.θ))
        return [da, de, di, dω, dΩ, dθ]
    end

    function Base.collect(oe::OE)::Array
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)
        return [a, e, i, ω, Ω, θ]
    end

    begin
        Velocity(oe::OE) = Velocity(oe.a, oe.r)
        AngularVelocity(oe::OE) = AngularVelocity(oe.a)
        TimePeriod(oe::OE) = TimePeriod(oe.a)
        Eccentricity(oe::OE) = oe.e
        rₚ(oe::OE) = rₚ(oe.a, oe.e)
    end
end

# ╔═╡ 1c921598-0b1b-4469-909f-8a3705d5463c
begin
    """
 The functions below characterize the senstivity of an Orbital Element to forces

 For example dΩ takes:
 oe : Orbital Elements 

 And returns a function to compute dΩ/dt upon application of force
 fᵣ : Force along radial direction
 fᵥ : Force along tangential direction
 fₕ : Force along angular momentum direction
 """

    function da(oe) # Semi Major Axis Senstivity
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

        return (fᵣ, fᵥ, fₕ) -> (2 * a^2 / h) * (e * sin(θ) * fᵣ + (p / r) * fᵥ)
    end

    function de(oe) # Eccentricity Senstivity
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

        return (fᵣ, fᵥ, fₕ) -> (1 / h) * (p * sin(θ) * fᵣ + ((p + r) * cos(θ) + r * e) * fᵥ)
    end

    function di(oe) # Inclination Senstivity
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

        return (fᵣ, fᵥ, fₕ) -> ((r * cos(θ + ω)) / h) * fₕ
    end

    function dω(oe) # Argument of periapsis Senstivity
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

        return (fᵣ, fᵥ, fₕ) -> (-p * cos(θ) * fᵣ + (p + r) * sin(θ) * fᵥ) / (e * h) - (r * sin(θ + ω) * cos(i) * fₕ) / (h * sin(i))
    end

    function dΩ(oe) # RAAN Senstivity
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

        return (fᵣ, fᵥ, fₕ) -> (r * sin(θ + ω)) / (h * sin(i)) * fₕ
    end

    function dθ(oe) # True anomaly Senstivity
        (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

        return (fᵣ, fᵥ, fₕ) -> h / r^2 + ((p * cos(θ) * fᵣ) - ((p + r) * sin(θ) * fᵥ)) / (e * h)
    end

    function δoe(oe::OE) # Infinitesimal senstivity of OEs at given state
        return ([da, de, di, dω, dΩ, dθ] .|> x -> x(oe))
    end

    function Δoe(oe::OE, F=[0, 0, 0], t=1) # Total change of OEs for given time t and force F
        return δoe(oe) .|> x -> x(F...) * t
    end
end

# ╔═╡ e1cde442-94b4-4bfb-84c4-4e6d0e3a0573
begin
    """
 Manoeuver is defined as going from an initial set of orbital elements oeᵢ to target set of orbital elements oeₜ. 
 Further a time limit on by when this manoeuver should be completed is imposed.

 In case of urgent and immediate change, tₑ = 0.
 The default constructor sets tₑ to TimePeriod of Orbit to account for optimal spot to take the manoeuver.
 """
    struct Manoeuver
        oeᵣ::OE # Current State
        oeₜ::OE # Target State
        tₑ::Real # By when manoeuver should be completed
    end
    function Manoeuver(oeᵢ, oeₜ)
        Manoeuver(oeᵢ, oeₜ, TimePeriod(oeᵢ))
    end
end

# ╔═╡ 1db35080-b837-4845-a49e-2a81a44938c1
function MaxΔOE(oe::OE) # Maximum change in OEs possible for a given OE
    (a, e, i, ω, Ω, θ, p, r, h) = (oe.a, oe.e, oe.i, oe.ω, oe.Ω, oe.θ, oe.p, oe.r, oe.h)

    da = 2 * √((a^3 * (1 + e)) / (μ * (1 - e)))
    de = 2 * p / h
    di = p / (h * (√(1 - e^2 * sin(ω)^2) - e * abs(cos(ω)))) |> x -> isinf(x) ? 0 : x
    dΩ = p / (h * sin(i) * (√(1 - e^2 * sin(ω)^2) - e * abs(sin(ω)))) |> x -> isinf(x) ? 0 : x
    dω = dΩ * abs(cos(i)) |> x -> isinf(x) ? 0 : x
    dθ = √(p^2 + (p + r)^2) / (e * h) + h / r^2 |> x -> isinf(x) ? 0 : x
    return [da, de, di, dΩ, dω, dθ]
end

# ╔═╡ bb8e8c30-b5ac-4d82-8b71-c9610673e3f9
function ThrustAngles(α, β)
    fᵣ = cos(β) * sin(α)
    fᵥ = cos(β) * cos(α)
    fₕ = sin(β)

    return [fᵣ, fᵥ, fₕ]
end

# ╔═╡ 3303a5ba-3a12-40f0-a37d-a46fae2c37fb
function ComputeForceDirectionFireTime(m::Manoeuver)
    OEDiff = m.oeₜ - m.oeᵣ
    function PercentErrorOE(oe1::OE, oe2::OE)
        oe1 = collect(oe1)
        oe2 = collect(oe2)
        (oe1 - oe2) ./ [oe1[1], 1, 2π, 2π, 2π, 2π] .|> abs
    end
    Perror = PercentErrorOE(m.oeᵣ, m.oeₜ)

    function OptimDirection(D) # Find optimal direction of impulse
        (α, β) = D
        F = ThrustAngles(α, β)

        Perturbation = Δoe(m.oeᵣ, F, 1)
        Perturbation = Perturbation

        priority = Perror #|> x->(x/maximum(x)).^Inf
        priority[[3,4]] .= 0e-3 # i and ω
        priority[[5,6]] .= [0e-3,0e-2] # Ω and θ
        DiffDirection = OEDiff .* priority

        return -(Perturbation' * DiffDirection)
    end

    D = optimize(OptimDirection, [0.0, 0.0], Newton(), autodiff=:forward)
    # Convert LVLH angles to direction
    F = ThrustAngles(D.minimizer...)
    # Find thruster fire time in seconds
    t = Perror * norm(Δoe(m.oeᵣ, F, 1)) / norm(MaxΔOE(m.oeᵣ)) |> norm

    pow = 1e-1
    return (F, ((((t + pow) / pow)^pow) - 1) / pow)
end

# ╔═╡ 2baa0fab-e20a-4304-ab05-c52060d62418
function LVLH2ECI(u, eci)
    iᵣ = eci[1:3] |> x -> x / norm(x)
    iᵥ = eci[4:6] |> x -> x / norm(x)
    iₕ = cross(iᵣ, iᵥ) |> x -> x / norm(x)
    return [iᵣ iᵥ iₕ] * u
end

# ╔═╡ 4a1fb6d9-25fa-4320-8d79-b27bb6ded0c3
let epi = Epoch(2020, 1, 1), days = 2, epf = epi + 60

    # For ground based manoeuvres
    currentState = [R_EARTH + 500e3, 1e-4, π / 2, π / 3, π / 4, π / 3]
    oeᵣ = OE(currentState) # Real OEs
    eci = sOSCtoCART(currentState)

    targetState = [R_EARTH + 600e3, 1e-4, π / 2, π / 3, π / 4, π / 3]
    oeₜ = OE(targetState) # Target OEs
    ecit = sOSCtoCART(targetState)

    impulses = floor(Int, (days * 86400) / (epf - epi))
    impulseGiven = Matrix{Float64}(undef, impulses, 3)
    impulseTime = Matrix{Float64}(undef, impulses, 1)
    c_OEs = Matrix{Float64}(undef, impulses, 6)
    t_OEs = Matrix{Float64}(undef, impulses, 6)

    satMass = 1 # kg
    thruster = 1 # N
    currentOrbit = EarthInertialState(epi, eci, dt=epf - epi,
        area_drag=1.0, coef_drag=1,
        area_srp=1.0, coef_srp=1,
        mass=satMass, n_grav=10, m_grav=10,
        drag=false, srp=false,
        moon=false, sun=false,
        relativity=false)
    targetOrbit = EarthInertialState(epi, ecit, dt=epf - epi,
        area_drag=.0, coef_drag=0,
        area_srp=0.0, coef_srp=0,
        mass=satMass, n_grav=1, m_grav=1,
        drag=false, srp=false,
        moon=false, sun=false,
        relativity=false)
    dt = epf - epi

    for i = 1:impulses
        oeᵣ = OE(sCARTtoOSC(eci)) # Real OEs
        c_OEs[i,:] = collect(oeᵣ)
        oeₜ = OE(sCARTtoOSC(ecit)) # Target OEs
        t_OEs[i,:] = collect(oeₜ)

        F, firingTime = ComputeForceDirectionFireTime(Manoeuver(oeᵣ, oeₜ))
        firingTime = isnan(firingTime) ? Inf : firingTime
        firingTime = min(firingTime, 1) # Maximum duration of fire is 1 sec
        ΔV = LVLH2ECI(F, eci) * firingTime # F[LVLH] -> F[ECI], F[ECI] * t
        currentOrbit.x[4:6] += ΔV * thruster / satMass

        impulseGiven[i, :] = F #LVLH Frame
        impulseTime[i] = firingTime

        step!(currentOrbit, dt)
        eci = currentOrbit.x
        step!(targetOrbit, dt)
        ecit = targetOrbit.x
    end
    @info "Total time fired" sum(impulseTime)
    diff = Matrix{Float64}(undef, impulses, 6)
    diff = t_OEs - c_OEs

    dayRange = range(0, days, impulses)
    fig = Figure(size=(900, 900))

    a = Axis(fig[1, 1],
        xlabel="Days", ylabel="Δa [m]",# yscale=Makie.pseudolog10,
        title="OEs with correction")
    e = Axis(fig[1, 1],
        ylabel="Δe", yaxisposition=:right)
    lines!(a, dayRange, diff[:, 1], label="a", color=RGBAf(0.8, 0.1, 0.1, 0.5))
    lines!(e, dayRange, diff[:, 2], label="e", color=RGBAf(0.1, 0.8, 0.1, 0.5))
    axislegend(a, position=:lt)
    axislegend(e, position=:rt)
    #    text!(a, days / 2, 0,
    #        text="Initial = [500e3, 0, π/2, π/4, π/4, 0]\nTarget = [600e3, 0, π/2, π/4, π/4, 0]",
    #        align=(:center, :bottom))

    deg = Axis(fig[2, 1],
        xlabel="Days", ylabel="Δ rad",
        title="i and ω variation")
    lines!(deg, dayRange, diff[:, 3], label="i", color=RGBAf(1,0.3,0.3,0.5))
    lines!(deg, dayRange, diff[:, 4], label="ω", color=RGBAf(0,1.3,0.3,0.5))
    lines!(deg, dayRange, diff[:, 5], label="Ω", color=RGBAf(0,0.3,1.3,0.5))
    lines!(deg, dayRange, diff[:, 6], label="θ", color=RGBAf(0,0.3,0.3,0.5))
    axislegend(deg, position=:lt)

    v = Axis(fig[3, 1],
        ylabel="ΔV Direction", xlabel="Days",
        title="Impulse Magnitude and Direction")
    t = Axis(fig[3, 1],
        ylabel="Firing Time [s]", yaxisposition=:right)
    series!(v, dayRange, impulseGiven',
        linewidth=0.1, markersize=3, color=:Set1, strokecolor=:white, alpha=0.1,
        labels=["iᵣ" "iᵥ" "iₕ"])
    lines!(t, dayRange, impulseTime[:], label="Firing Time", color=RGBAf(1, 0.3, 0.3, 0.5))
    axislegend(v, position=:rt)
    axislegend(t, position=:rb)
    linkyaxes!(v,t)
    linkxaxes!(a,e)
    linkxaxes!(a,v)
    linkxaxes!(a,t)
    linkxaxes!(a,deg)
    #    text!(v, 0, 0,
    #	  text = "Firing interval=1m\nFiring Time Limit=1s\nSatMass=$satMass kg\nT_Str=$thruster N",
    #	  align = (:left, :bottom))
    fig
end