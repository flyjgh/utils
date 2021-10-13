using Random, CUDA, Flux

d1 = Dense2(2048, 1024)
d2 = Dense3(2048, 1024)
d3 = Dense4(2048, 1024)
ind = rand(Float32, 2048, 64) |> CuArray

f(NN, x::T) where T = NN(x)
f(d3, ind)

using BenchmarkTools
@benchmark f($(Dense2(2048, 1024)),$(rand(Float32, 2048, 64)))
@benchmark f($(Dense3(2048, 1024)),$(rand(Float32, 2048, 64) |> CuArray))
@benchmark f($(Dense4(2048, 1024)),$(rand(Float32, 2048, 64) |> CuArray))
using Flux: σ
