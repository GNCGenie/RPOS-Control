begin
    using LinearAlgebra
    using GLMakie
    using SatelliteDynamics
end

function ECItoLVLH(state)
    r = state[1:3]
    v = state[4:6]

    ir = normalize(r)
    ih = normalize(r × v) |> x->((!isnan).(x)).*x
    iv = normalize(ih × ir) |> x->((!isnan).(x)).*x
    R = [ir iv ih]
    return [R zeros(3, 3)
            zeros(3, 3) R]
end

begin
    struct State{T}
        x::T
        ẋ::T
        r::T
        ṙ::T
    end
    State(args...) = State(promote(args...)...)

    struct TrajectoryLimiter{T}
        Ts::T
        ẋM::T
        ẍM::T
    end
    TrajectoryLimiter(args...) = TrajectoryLimiter(promote(args...)...)

    function (limiter::TrajectoryLimiter)(state, r::Number)
        trajlim(state, r, limiter.Ts, limiter.ẋM, limiter.ẍM)
    end
    function trajlim(state, rt, T, ẋM, ẍM)
        sat(x) = begin
            λ = 1e-1
            x = λ * x
            clamp(x, -one(x), one(x))
        end

        # Update states for control
        (; x, ẋ, r, ṙ) = state
        TU = T * ẍM
        ṙ = (rt - r) / T
        r = rt

        # Calculate control input according to state
        effort(x, ẋ, r, ṙ) = begin
            y = x - r
            ẏ = ẋ - ṙ

            z = 1 / TU * (y / T + ẏ / 2)
            ż = ẏ / TU
            m = floor((1 + √(1 + 8abs(z))) / 2)
            σ = ż + z / m + (m - 1) / 2 * sign(z)
            u = -ẍM * sat(σ) * (1 + sign(ẋ * sign(σ) + ẋM - TU)) / 2
            return u
        end
        u = effort(x, ẋ, r, ṙ)

        ẋ1 = T * u + ẋ
        x1 = T / 2 * (ẋ1 + ẋ) + x

        State(x1, ẋ1, r, ṙ), u
    end
end

begin
    epc0 = Epoch(2020, 1, 1, 0, 0, 0, 0.0)
    oe = [R_EARTH + 650e3, 0.0, 0.0, 0.0, 0.0, 0.0]

    params = (dt=10.0, area_drag=1e0, coef_drag=1,
              area_srp=1e0, coef_srp=1,
              mass=100, n_grav=10, m_grav=10,
              drag=false, srp=false,
              moon=false, sun=false,
              relativity=false)
    ecit = sOSCtoCART(oe, use_degrees=true)
    orbt = EarthInertialState(epc0, ecit; params...)
    ecic = ecit - [1e3, 1e3, 1e3, 0e1, 0e1, 0e1]
    orbc = EarthInertialState(epc0, ecic; params...)

    function simStep!(orbc, orbt, dt=10.0) # Simulate both sats ahead by dt
        step!(orbc, dt)
        step!(orbt, dt)
        return orbc, orbt
    end
    function genStates(orbc, orbt) # Generate states for control
        return [State(orbc.x[i] - orbt.x[i], orbc.x[i+3] - orbt.x[i+3], 0.0, 0.0) for i = 1:3]
    end
end

let orbc = orbc, orbt = orbt
    timeStep = 1e0
    limiter = TrajectoryLimiter(timeStep, 1e1, 1e-1)

    # Empty arrays for storing simulation data
    trc = Vector{Vector{Float64}}(undef, 0)
    trt = Vector{Vector{Float64}}(undef, 0)
    effort = Vector{Vector{Float64}}(undef, 0)
    totalFiringTime = 0.0

    ########################################
    # Simulation Loop
    ########################################
    num_time_steps = 2^10
    for i = 1:num_time_steps
        # Generate Control input
        states = genStates(orbc, orbt)
        u = map(state -> limiter(state, 0.0)[2], states)
        orbc.x[4:6] += u
        totalFiringTime += u .|> abs |> sum

        # Simulate both satellites
        orbc, orbt = simStep!(orbc, orbt, timeStep)
        push!(trc, ECItoLVLH(orbt.x) * orbc.x)
        push!(trt, ECItoLVLH(orbt.x) * orbt.x)
        push!(effort, u)

        if norm(trc[end] - trt[end]) < 1
            println((t -> "$(t÷60)m$(t%60)s")(i * timeStep)) # Print time taken to reach as XmYs
            println(round(totalFiringTime; digits=2)) # Total firing duration
            break
        end
    end

    ########################################
    # Printing utility for simulation
    ########################################
    f = Figure(; size=(900, 1200))
    ax = Axis3(f[:, :], perspectiveness=0.3, xlabel="x(LVLH) [m]", ylabel="y(LVLH) [m]", zlabel="z(LVLH) [m]")
    scatter!(ax, (trt[begin]-trc[begin])[1:3]...; label="Chaser",)
    scatter!(ax, zeros(3)...; label="Target")
    [Point3((t-c)[1:3]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax, x; color=range(0, 1, length(x)))
    axislegend(ax)

    min_range = range(0, length(trc)/60, length(trc))
    ax2 = Axis(f[2, 1], xlabel="Minutes", ylabel="Distance [m]")
    ax2r = Axis(f[2, 1], xlabel="Minutes", ylabel="Velocity [m/s]", yaxisposition=:right)
    [norm((t-c)[1:3]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2, min_range, x; color=:red, label="position")
    [norm((t-c)[4:6]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2r, min_range, x; color=:green, label="velocity")
    axislegend(ax2, position=:lt)
    axislegend(ax2r, position=:rt)
    linkxaxes!(ax2, ax2r)

    ax3 = Axis(f[3, 1], ylabel="ΔV [m/s]", xlabel="Minutes")
    reduce(hcat, effort) |> x -> series!(ax3, min_range, x; labels=["r" "v" "h"])
    axislegend(ax3)


    #text!(0,0,text=string("ΔV = ", round(totalFiringTime;digits=2), "m/s"))
    @info string("ΔV = ", round(totalFiringTime;digits=2), "m/s")

    f
end
