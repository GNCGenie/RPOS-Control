begin
    using JuMP
    using Ipopt
    using LinearAlgebra
end

begin
    model = Model(Ipopt.Optimizer)

    # Define our constant parameters
    Δt = 1e0
    num_time_steps = 10^3
    max_position = 1e4
    max_velocity = 1e1
    max_effort = 1e-1

    R_Earth = 6.3781363e6
    GM_Earth = 3.986004415e14
    a = R_Earth + 650e3
    n = √(GM_Earth/a^3)

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
                position[j, i] == position[j, i-1] + velocity[j, i-1] * Δt + effort[j, i-1] * Δt^2 / 2)

    # Cost function: minimize final position and final velocity
    @objective(model, Min, sum((effort[:, :]).^2))

    # Initial conditions:
    @constraint(model, position[:, begin] .== [5e1, 5e1, 5e1])
    @constraint(model, velocity[:, begin] .== [0e0, 0e0, 0e0])

    # Final conditions:
    @constraint(model, position[:, end] .== [0e1, 1e1, 0e1])
    @constraint(model, velocity[:, end] .== [0e0, 0e0, 0e0])
end
JuMP.optimize!(model)

using GLMakie
let
    fig = Figure(size=(1920, 1080))

    ax1 = LScene(fig[1, 1], scenekw = (backgroundcolor=RGBf(0.3,0.3,0.1), clear=true))
    # ax1 = Axis3(fig[1, 1], perspectiveness=0.5, xlabel="r-bar", ylabel="v-bar", zlabel="h-bar")

    scatter!(ax1, [0,0,0]'; label="Target")
    scatter!(ax1, value.(position[:,begin])'; label="Chaser")
    strength = (JuMP.value.(effort)) |> eachcol .|> norm

    JuMP.value.(position) |> eachrow |> x -> lines!(ax1, x...; colormap=:buda, color=range(0,1,length(x)))
    arrows!(ax1, (JuMP.value.(position) |> eachrow)..., (JuMP.value.(effort) |> eachrow)...,
            arrowsize=5e-2, lengthscale=1e4, colormap=:thermal,
            arrowcolor=strength, linecolor=strength)
    axislegend(ax1)
    @info value.(effort) .|> abs |> sum
    fig
end
