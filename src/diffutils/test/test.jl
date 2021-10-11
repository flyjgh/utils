include("../src/diffutils.jl")
using .diffutils
using Statistics
using Flux: Dense, ADAM, relu, gradient, params, update!


m2 = Parallelism(Dense(4, 2) ∘ vcat, Nonlinear(2), Nonlinear(2))

function loss(model, X, y)
    ŷ = model(X)
    return (ŷ .- y) .^ 2 |> mean
end

μ = 3f-4
optimiser = ADAM(μ)

function trainstep(model, X, y, opt)
    ∂ = gradient(() -> loss(model, X, y), params(model))
    update!(opt, params(model), ∂)
    return loss(model, X, y)
end

function train(model, opt, n)
    for _ ∈ 1:n
        X = rand(Float32, 2, 100)
        y = X .> 5f-1 .|> Float32
        @show trainstep(model, X, y, opt)
    end
end

train(m2, optimiser, 10000)

m2([4f-1, 6f-1])
n = Nonlinear(2)
