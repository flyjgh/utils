include("../src/diffutils.jl")
using .diffutils
using Statistics
using Flux: ADAM, Chain, gradient, params, update!


m  = Parallelism(Nonlinear(4, 2) ∘ vcat, Nonlinear(2), Nonlinear(2));
m1 = WCell(Nonlinear(2), Nonlinear(2));
m2 = CWCell(Nonlinear(4,1), Nonlinear(2));
m3 = Chain(Parallelism(.*, m2,m2,m2,m2), Parallelism(.*, m2,m2,m2,m2), Parallelism(.*, m2,m2,m2,m2), Parallelism(.*, m2,m2,m2,m2));

function loss(model, X, y)
    ŷ = model(X)
    return (ŷ .- y) .^ 2 |> mean
end

opt = ADAM(3f-4)

function trainstep(model, X, y, opt)
    ∂ = gradient(() -> loss(model, X, y), params(model))
    update!(opt, params(model), ∂)
    return loss(model, X, y)
end

function separate(model, opt, n)
    for _ ∈ 1:n
        X = rand(Float32, 2, 100)
        y = (X .> 5f-1) .|> Float32
        @show trainstep(model, X, y, opt)
    end
end

function dsin(model, opt, n)
    for _ ∈ 1:n
        X = rand(Float32, 2, 100)
        y = sin.(X .* 2f0)
        @show trainstep(model, X, y, opt)
    end
end

function mixeq(model, opt, n)
    for _ ∈ 1:n
        X = rand(Float32, 2, 100)
        y = sin.(8f0exp.(X) .* 8f0X.^exp.(X)) ./ 8f0tan.(X .%4f0)
        @show trainstep(model, X, y, opt)
    end
end

dsin(m2, optimiser, 10000)
m2([4f-1 6f-1
    6f-1 4f-1])

x = rand(Float32, 2, 4)
sin.(8f0exp.(x) .* 8f0x.^exp.(x)) ./ 8f0tan.(x .%4f0)
m(x)
