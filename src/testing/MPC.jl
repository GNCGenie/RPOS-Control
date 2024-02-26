begin
    using JuMP
    using SCS
    using Ipopt
    using LinearAlgebra
    using GLMakie
    using UnicodePlots
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

function dX(X, U)
    R_Earth = 6.3781363e6
    GM_Earth = 3.986004415e14
    a = R_Earth + 650e3
    n = √(GM_Earth / a^3)

    A = [0 0 0 1 0 0
         0 0 0 0 1 0
         0 0 0 0 0 1
         3*n^2 0 0 0 2*n 0
         0 0 0 -2*n 0 0
         0 0 -n^2 0 0 0]
    B = [0 0 0
         0 0 0
         0 0 0
         1 0 0
         0 1 0
         0 0 1]
    return A*X + B*U
end

function mpcGetEffort(initState::Vector, targetState::Vector, dt=1e0::Float64)

    model = Model(SCS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iters", 10)

    rel_pos = initState[1:3] - targetState[1:3] |> norm
    rel_vel = initState[4:6] - targetState[4:6] |> norm

    # Define our decision variables
    horizon = 20
    @variables model begin
        X[1:6, 1:horizon+1]
        U[1:3, 1:horizon]
    end

    # Define our constant parameters
    max_velocity = 5e0
    max_effort = 1e-1
    dt = max(rel_pos/(1e0*horizon) , 1e-1)
    # Add dynamics constraints
    for i = 1:horizon
        k1 = dX(X[:, i], U[:, i])
        k2 = dX(X[:, i] + 0.5*dt*k1, U[:, i])
        k3 = dX(X[:, i] + 0.5*dt*k2, U[:, i])
        k4 = dX(X[:, i] + dt*k3, U[:, i])
        @constraint(model, X[:, i+1] .== X[:, i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
    end
    @constraint(model, [i = 1:horizon],
                -max_velocity .<= X[4:6, i] .<= max_velocity)
    @constraint(model, [i = 1:horizon],
                -max_effort*dt .<= U[:, i] .<= max_effort*dt)

    # Boundary conditions:
    @constraint(model, X[:, begin] .== initState)
    @constraint(model, X[:, end] .== targetState)
    @constraint(model, U[:, end] .== 0)
    @objective(model, Min, sum(U[:, :].^2))

    JuMP.optimize!(model)
    effort = JuMP.value.(U)
    plt = lineplot(effort[1, :]; height=40, width=150, ylim=[-0.1, 0.1])
    lineplot!(plt, effort[2, :])
    lineplot!(plt, effort[3, :])
    println(targetState - initState)
    println(targetState - JuMP.value.(X[:, end]))
    display(plt)
    return effort[:, begin]
end

begin
    initialState = [0e3, 0.5e3, 0, 0, 0, 0]
    targetState = [0, 1.0e3, 0, 0, 0, 0]
    epc0 = Epoch(2020, 1, 1, 0, 0, 0, 0.0)
    oe = [R_EARTH + 650e3, 0.0, 0.0, 0.0, 0.0, 0.0]

    params = (dt=1e0, area_drag=1e0, coef_drag=1e0,
              area_srp=1e0, coef_srp=1e0,
              mass=100, n_grav=3, m_grav=3,
              drag=false, srp=false,
              moon=false, sun=false,
              relativity=false)
    ecit = sOSCtoCART(oe, use_degrees=true)
    orbt = EarthInertialState(epc0, ecit; params...)
    ecic = ecit - initialState
    orbc = EarthInertialState(epc0, ecic; params...)

    function simStep!(orbc, orbt, dt=1.0) # Simulate both sats ahead by dt
        step!(orbc, dt)
        step!(orbt, dt)
        return orbc, orbt
    end
end

let orbc = orbc, orbt = orbt
    # Empty arrays for storing simulation data
    trc = Vector{Vector{Float64}}(undef, 0)
    trt = Vector{Vector{Float64}}(undef, 0)
    effort = Vector{Vector{Float64}}(undef, 0)
    totalimpulse = 0.0

    ########################################
    # Simulation Loop
    ########################################
    timeStep = 1e0
    num_time_steps = 2^11
    for i = 1:num_time_steps
        # Generate Control input
        R = ECItoLVLH(orbt.x) # 6x6 Matrix for mapping ECI->LVLH
        lvlhc = -R * (orbt.x-orbc.x) # Position of chaser w.r.t. target in LVLH
        lvlht = zeros(6) # Target is always at origin
        u = mpcGetEffort(lvlhc, lvlht-targetState, timeStep) # Returns impulse in LVLH Frame
        u = inv(R[1:3,1:3]) * u # Convert LVLH to ECI for simulation
        u = clamp.(u, -1e-1, 1e-1)
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
    f = Figure(; size=(900, 600))

    ax = Axis3(f[:, :], perspectiveness=0.3, xlabel="R(LVLH) [m]", ylabel="V(LVLH) [m]", zlabel="H(LVLH) [m]")
    scatter!(ax, (trt[begin]-trc[begin])[1:3]...; label="Chaser",)
    scatter!(ax, zeros(3)...; label="Target")
    [Point3((ECItoLVLH(t)*(t-c))[1:3]) for (t,c) in zip(trt,trc)] |> x -> lines!(ax, x; color=range(0, 1, length(x)))
    axislegend(ax)

    min_range = range(0, num_time_steps/60, num_time_steps)
    ax2 = Axis(f[2, 1], xlabel="Minutes", ylabel="Distance [m]")
    ax2r = Axis(f[2, 1], xlabel="Minutes", ylabel="Velocity [m/s]", yaxisposition=:right)
    [norm((t-c)[1:3]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2, min_range, x; color=:red, label="position")
    [norm((t-c)[4:6]) for (c, t) in zip(trc, trt)] |> x -> lines!(ax2r, min_range, x; color=:green, label="velocity")
    axislegend(ax2, position=:lt)
    axislegend(ax2r, position=:rt)
    linkxaxes!(ax2, ax2r)
    #linkyaxes!(ax2, ax2r)

    ax3 = Axis(f[3, 1], ylabel="ΔV [m/s]")
    reduce(hcat, effort) |> x -> series!(ax3, min_range, x; labels=["r" "v" "h"])
    axislegend(ax3, position=:lt)
    linkxaxes!(ax2, ax3)

    @info "Total impulse used" totalimpulse
    f
end
