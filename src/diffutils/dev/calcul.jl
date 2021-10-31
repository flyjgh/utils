using Random
using Flux
using CUDA

include("../src/diffutils.jl")
using .diffutils

println("num ?")
Parallelism(.*, exp2, √)(parse(Float64, readline())) |> println
