using Zygote, Flux, Random
# using CUDA

include("../src/diffutils.jl")
using .diffutils

Parallelism(.*, exp2, √)(parse(Float64, readline())) |> println

model = Recurrent(vcat, Nonlinear(2,1) ∘ Nonlinear(1,2), 1)
m2 = Chain(Densegpu(512,256),Densegpu(256,128),Densegpu(128,256),Densegpu(256,512))
m3 = Chain(Dense(512,256),Dense(256,128),Dense(128,256),Dense(256,512)) |> gpu

using BenchmarkTools
v= Float32 - (512,512) |> gpu

function test(m,v,n)
    for _ ∈ 1:n m(v) |> gpu end
end

@benchmark test($m2,$v,100)
@benchmark test($m3,$v,100)

r = 1f0:100f0
suites = [r .+ 2, r .* 2.3, r .^ 2, .√r, log.(r), r .* .√r]

function loss(model, X, y)
    ŷ = model(X)
    return (ŷ .- y) .^ 2 |> mean
end

function trainstep(model, X, y, opt)
    ∂ = gradient(() -> loss(model, X, y), params(model))
    update!(opt, params(model), ∂)
    return loss(model, X, y)
end

function train(model, n)
    opt = ADAM(3f-4)
    for _ ∈ 1:n
        X = rand(Float32, 2, 100)
        y = (X .> 5f-1) .|> Float32
        @show trainstep(model, X, y, opt)
    end
end
