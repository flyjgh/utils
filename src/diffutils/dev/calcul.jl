using Random, CUDA
using Flux: @functor, update!

nfan() = 1, 1
nfan(n) = 1, n
nfan(out, in) = in, out
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])  # Conv Kernel
glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 5f-1) .* sqrt(24f0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(Random.GLOBAL_RNG, dims...)

struct Chain{T <: Tuple}
    layers::T
end
Chain(layers...) = Chain(layers)

@generated function apply_chain(c::T, x) where {T}
    foldl(1:length(T.parameters), init=:x) do b, i
        :(c[$i]($b))
    end
end
(c::Chain)(x) = apply_chain(c.layers, x)

struct Densegpu{A}
    W::CuMatrix{Float32}
    b::CuVector{Float32}
    σ::A
end

function Densegpu(in, out, σ=identity)
    Densegpu(glorot_uniform(out, in) |> CuArray, glorot_uniform(out) |> CuArray, σ)
end
@functor Densegpu

(a::Densegpu)(x::CuArray{Float32,N}) where {Float32,N} = a.σ(a.W * x .+ a.b)

d = Densegpu(2048, 1024)
ind = rand(Float32, 2048, 64) |> CuArray

f(NN, x::T) where T = NN(x)

using BenchmarkTools
@benchmark f(d, ind)

CuMatrix(3)
