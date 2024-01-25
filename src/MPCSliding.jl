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
    ih = normalize(r × v) |> x->((!isnan).(x)).*x
    iv = normalize(ih × ir) |> x->((!isnan).(x)).*x
    R = [ir iv ih]
    return [R zeros(3, 3)
            zeros(3, 3) R]
end

function mpcGetEffort(initState::Vector, targetState::Vector)

    R_Earth = 6.3781363e6
    GM_Earth = 3.986004415e14
    a = R_Earth + 650e3
    n = √(GM_Earth / a^3)

    rel_pos = initState[1:3] - targetState[1:3] |> norm
    rel_vel = initState[4:6] - targetState[4:6] |> norm

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Define our constant parameters
    num_time_steps = 5
    Δt = 1e0
    max_position = 1e4
    max_effort = 1e-1
    max_velocity = sqrt(rel_pos*max_effort)/(Δt*num_time_steps) + 1e-1

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
    @constraint(model, [i = 2:num_time_steps, j = 1:3],
                effort[j, i] .== effort[j, 1]
                )

    # Cost function: minimize final position and final velocity
    @objective(model, Min,
               sum((position[:, end] - targetState[1:3]) .^ 2)
               + sum((velocity[:, end] - targetState[4:6]) .^ 2)
              )

    JuMP.optimize!(model)
    results = JuMP.value.(effort)
    return results[:, begin]
end

begin
    epc0 = Epoch(2020, 1, 1, 0, 0, 0, 0.0)
    oe = [R_EARTH + 650e3, 0.0, 0.0, 0.0, 0.0, 0.0]

    params = (dt=1e0, area_drag=1e0, coef_drag=1e0,
              area_srp=1e0, coef_srp=1e0,
              mass=100, n_grav=9, m_grav=9,
              drag=false, srp=false,
              moon=false, sun=false,
              relativity=false)
    ecit = sOSCtoCART(oe, use_degrees=true)
    orbt = EarthInertialState(epc0, ecit; params...)
    ecic = ecit - [1e3, 1e3, 1e3, 0e1, 0e1, 0e1]
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

    ########################################
    # Simulation Loop
    ########################################
    num_time_steps = 2^10
    for i = 1:num_time_steps
        # Generate Control input
        R = ECItoLVLH(orbt.x) # 6x6 Matrix for mapping ECI->LVLH
        lvlhc = -R * (orbt.x-orbc.x) # Position of chaser w.r.t. target in LVLH
        lvlht = zeros(6) # Target is always at origin
        u = mpcGetEffort(lvlhc, lvlht-[0,10,0,0,0,0]) # Returns impulse in LVLH Frame
        u = inv(R[1:3,1:3]) * u # Convert LVLH to ECI for simulation
        orbc.x[4:6] += u # Add impulse velocity to simulation velocity
        totalimpulse += norm(u)
        push!(effort, u)

        # Simulate both satellites
        orbc, orbt = simStep!(orbc, orbt, timeStep)
        push!(trc, orbc.x)
        push!(trt, orbt.x)
    end

    ########################################
    # Printing utility for simulation
    ########################################
    f = Figure(; size=(900, 1200))

    ax = Axis3(f[:, :], perspectiveness=0.3, xlabel="x(LVLH) [m]", ylabel="y(LVLH) [m]", zlabel="z(LVLH) [m]")
    scatter!(ax, (trt[begin]-trc[begin])[1:3]...; label="Chaser",)
    scatter!(ax, zeros(3)...; label="Target")
    [Point3((ECItoLVLH(t)*(t-c))[1:3]) for (t,c) in zip(trt,trc)] |> x -> lines!(ax, x; color=range(0, 1, length(x)))
    axislegend(ax)

    min_range = range(0, num_time_steps/60, num_time_steps)
    ax2 = Axis(f[2, 1], xlabel="Minutes", ylabel="Distance [m]")
    ax2r = Axis(f[2, 1], xlabel="Minutes", ylabel="Velocity [m]", yaxisposition=:right)
    [norm((t-c)[1:3]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2, min_range, x; color=:red, label="position")
    [norm((t-c)[4:6]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2r, min_range, x; color=:green, label="velocity")
    axislegend(ax2, position=:lt)
    axislegend(ax2r, position=:rt)
    linkxaxes!(ax2, ax2r)
    linkyaxes!(ax2, ax2r)

    ax3 = Axis(f[3, 1])
    reduce(hcat, effort) |> x -> series!(ax3, min_range, x; labels=["r" "v" "h"])
    axislegend(ax3)
    linkxaxes!(ax2, ax3)

    @info "Total impulse used" totalimpulse
    f
end
