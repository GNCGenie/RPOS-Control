begin
    using JuMP
    using Ipopt
    using LinearAlgebra
end

begin
    model = Model(Ipopt.Optimizer)
    R_Earth = 6.3781363e6
    GM_Earth = 3.986004415e14
    a = R_Earth + 650e3
    n = √(GM_Earth/a^3)

    # Define our constant parameters
    Δt = 1e0
    num_time_steps = 2400
    max_position = 1e4
    max_velocity = 1e1
    max_effort = 1e-1

    # Define our decision variables
    @variables model begin
        -max_position <= position[1:3, 1:num_time_steps] <= max_position
        -max_velocity <= velocity[1:3, 1:num_time_steps] <= max_velocity
        -max_effort <= effort[1:3, 1:num_time_steps] <= max_effort
    end

    # Add dynamics constraints
    @constraint(model, [i = 2:num_time_steps],
                velocity[1, i] == velocity[1, i-1] + 3*n^2*position[1,i-1] + 2*n*velocity[2,i-1] + effort[1, i-1] * Δt)
    @constraint(model, [i = 2:num_time_steps],
                velocity[2, i] == velocity[2, i-1] - 2*n*velocity[1,i-1] + effort[2, i-1] * Δt)
    @constraint(model, [i = 2:num_time_steps],
                velocity[3, i] == velocity[3, i-1] - n^2*position[3,i-1] + effort[3, i-1] * Δt)
    @constraint(model, [i = 2:num_time_steps, j = 1:3],
                position[j, i] == position[j, i-1] + velocity[j, i-1] * Δt) #  + effort[j, i-1] * Δt^2 / 2

    # Cost function: minimize final position and final velocity
    @objective(model, Min, sum(effort[:, :].^2))

    # Initial conditions:
#    @constraint(model, position[:, begin] .== [1e1, 0, 0])
#    @constraint(model, position[:, begin] .== [0, 2.5e3, 0])
    @constraint(model, position[:, begin] .== [0, 2e2, 0])
    @constraint(model, velocity[:, begin] .== [0e0, 0e0, 0e0])

    # Final conditions:
#    @constraint(model, position[:, end] .== [0, 2.5e3, 0])
#    @constraint(model, position[:, end] .== [0, 2e2, 0])
    @constraint(model, position[:, end] .== [0, 1e1, 0])
    @constraint(model, velocity[:, end] .== [0e0, 0e0, 0e0])
end
JuMP.optimize!(model)

using GLMakie
begin
    f = Figure()

    ax1 = Axis3(f[:, :], perspectiveness=0.3, xlabel="R(LVLH) [m]", ylabel="V(LVLH) [m]", zlabel="H(LVLH) [m]")
    xlims!(ax1, minimum(JuMP.value.(position)), maximum(JuMP.value.(position)))
    ylims!(ax1, minimum(JuMP.value.(position)), maximum(JuMP.value.(position)))
    zlims!(ax1, minimum(JuMP.value.(position)), maximum(JuMP.value.(position)))

    scatter!(ax1, [0,0,0]'; label="Target")
    scatter!(ax1, value.(position[:,begin])'; label="Chaser")
    strength = (JuMP.value.(effort)) |> eachcol .|> norm

    JuMP.value.(position) |> eachrow |> x -> lines!(ax1, x...; colormap=:buda, color=range(0,1,length(x)))
    arrows!(ax1, (JuMP.value.(position) |> eachrow)..., (JuMP.value.(effort) |> eachrow)...,
            arrowsize=5e-2, lengthscale=1e4, colormap=:thermal,
            arrowcolor=strength, linecolor=strength)
    axislegend(ax1)

    min_range = Δt*range(0, num_time_steps/60, num_time_steps)
    ax2 = Axis(f[2, 1], xlabel="Minutes", ylabel="Distance [m]")
    ax2r = Axis(f[2, 1], xlabel="Minutes", ylabel="Velocity [m/s]", yaxisposition=:right)
    JuMP.value.(position) |> eachcol .|> norm |> x->lines!(ax2, min_range, x; color=:red, label="Position")
    JuMP.value.(velocity) |> eachcol .|> norm |> x->lines!(ax2r, min_range, x; color=:green, label="Velocity")
    axislegend(ax2, position=:lt)
    axislegend(ax2r, position=:rt)

    ax3 = Axis(f[3, 1]; ylabel="ΔV [m/s]", xlabel="Minutes")
    JuMP.value.(effort) |> x -> series!(ax3, min_range, x; labels=["r" "v" "h"])
    axislegend(ax3, position=:lt)
    @info value.(effort) .|> abs |> sum
    f

    #   normed = eachcol(value.(effort)) .|> x->x/norm(x)
    #   angle = [normed[begin]' * i for i in normed] .|> x->acos(x) .|> rad2deg
    #   angle |> lines
    #   angle |> diff |> lines
    #    angle |> diff |> diff |> lines
end
