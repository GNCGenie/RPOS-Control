begin
    using JuMP
    using Ipopt
    using LinearAlgebra
    using GLMakie
    using SatelliteDynamics
end

function ECItoLVLH(state)
    r = state[1:3]
    v = state[4:6]

    ir = normalize(r)
    ih = normalize(r × v)
    iv = normalize(ih × ir)
    R = [ir iv ih]
    return R
    return [R zeros(3, 3)
        zeros(3, 3) R]
end

function mpcGetEffort(initState::Vector, targetState::Vector)
    rel_pos = initState[1:3] - targetState[1:3] |> norm
    rel_vel = initState[4:6] - targetState[4:6] |> norm

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Define our constant parameters
    num_time_steps = 2
    Δt = 1e0
    max_position = 1e4
    max_effort = 1e-1

#    sat(x) = begin
#        clamp(x, -one(x), one(x))
#    end
#    λ = 1e-0
#    k = √(max_effort/λ)
#    σ = rel_pos + (max(abs(rel_vel), k)*rel_vel)/(2*max_effort) + 3/(2*λ)*sat(rel_vel/k)
    max_velocity = sqrt(rel_pos*max_effort)/(Δt*num_time_steps) + 1e-1

    R_Earth = 6.3781363e6
    GM_Earth = 3.986004415e14
    a = R_Earth + 650e3
    n = √(GM_Earth / a^3)

    # Define our decision variables
    @variables model begin
        -max_position <= position[1:3, 1:num_time_steps] <= max_position
        -max_velocity <= velocity[1:3, 1:num_time_steps] <= max_velocity
        -max_effort <= effort[1:3, 1:num_time_steps] <= max_effort
    end

    # Initial conditions:
    @constraint(model, position[:, begin] .== initState[1:3])
    @constraint(model, velocity[:, begin] .== initState[4:6])

    # Add dynamics constraints
    @constraint(model, [i = 2:num_time_steps],
        velocity[1, i] == velocity[1, i-1] + 3 * n^2 * position[1, i-1] + 2 * n * velocity[2, i-1]
        + effort[1, i-1] * Δt
    )
    @constraint(model, [i = 2:num_time_steps],
        velocity[2, i] == velocity[2, i-1] - 2 * n * velocity[1, i-1]
        + effort[2, i-1] * Δt
    )
    @constraint(model, [i = 2:num_time_steps],
        velocity[3, i] == velocity[3, i-1] - n^2 * position[3, i-1]
        + effort[3, i-1] * Δt
    )
    @constraint(model, [i = 2:num_time_steps, j = 1:3],
        position[j, i] == position[j, i-1] + velocity[j, i-1] * Δt
        + effort[j, i-1] * Δt^2 / 2
    )

    # Cost function: minimize final position and final velocity
    @objective(model, Min,
        sum((position[:, end] - targetState[1:3]) .^ 2) +
        sum((velocity[:, end] - targetState[4:6]) .^ 2)
    )

    JuMP.optimize!(model)
    results = JuMP.value.(effort)
    return results[:, begin]
end

begin
    epc0 = Epoch(2020, 1, 1, 0, 0, 0, 0.0)
    oe = [R_EARTH + 400e3, 0.0, 0.0, 0.0, 0.0, 0.0]

    params = (dt=1e0, area_drag=1e0, coef_drag=1e0,
        area_srp=1e0, coef_srp=1e0,
        mass=100, n_grav=2, m_grav=2,
        drag=true, srp=true,
        moon=true, sun=true,
        relativity=false)
    ecit = sOSCtoCART(oe, use_degrees=true)
    orbt = EarthInertialState(epc0, ecit; params...)
    ecic = ecit - [5e1, 5e1, 5e1, 0e1, 0e1, 0e1]
    orbc = EarthInertialState(epc0, ecic; params...)

    function simStep!(orbc, orbt, dt=1.0) # Simulate both sats ahead by dt
        step!(orbc, dt)
        step!(orbt, dt)
        return orbc, orbt
    end
end

let orbc = orbc, orbt = orbt
    timeStep = 1e0

    # Empty arrays for storing simulation data
    trc = Vector{Vector{Float64}}(undef, 0)
    trt = Vector{Vector{Float64}}(undef, 0)
    effort = Vector{Vector{Float64}}(undef, 0)
    totalimpulse = 0.0
    push!(trc, orbc.x)
    push!(trt, orbt.x)

    ########################################
    # Simulation Loop
    ########################################
    for i = 1:2^11
        # Generate Control input
        R = ECItoLVLH(orbt.x)
        lvlhc = -[R * (orbt.x-orbc.x)[1:3]; R * (orbt.x-orbc.x)[4:6]]
        lvlht = zeros(6)
        u = mpcGetEffort(lvlhc, lvlht-[20,20,0,0,0,0])
        u = inv(R) * u
        orbc.x[4:6] += u
        totalimpulse += norm(u)

        # Simulate both satellites
        orbc, orbt = simStep!(orbc, orbt, timeStep)
        push!(trc, orbc.x)
        push!(trt, orbt.x)
        push!(effort, u)
    end

    ########################################
    # Printing utility for simulation
    ########################################
    f = Figure(; size=(900, 1200))

    ax = Axis3(f[:, :], perspectiveness=0.3, xlabel="x(ECI)", ylabel="y(ECI)", zlabel="z(ECI)")
    scatter!(ax, (trt[begin]-trc[begin])[1:3]...; label="Chaser",)
    scatter!(ax, zeros(3)...; label="Target")
    [Point3((t-c)[1:3]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax, x; color=range(0, 1, length(x)))
    axislegend(ax)

    ax2 = Axis(f[2, 1], xlabel="time", ylabel="dist wrt origin")
    ax2r = Axis(f[2, 1], xlabel="time", ylabel="vel wrt origin", yaxisposition=:right)
    [norm((t-c)[1:3]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2, x; color=range(0, 1, length(x)))
    [norm((t-c)[4:6]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2r, x; colormap=:thermal, color=range(0, 1, length(x)))
    #    axislegend(ax2)
    #    axislegend(ax2r)

    ax3 = Axis(f[3, 1])
    reduce(hcat, effort) |> x -> series!(ax3, x; labels=["r" "v" "h"])
    axislegend(ax3)

    linkxaxes!(ax2, ax3)
    linkxaxes!(ax2, ax2r)

    @info totalimpulse
    f
end