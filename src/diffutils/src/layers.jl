using Random
using CUDA: CuArray
using Flux: @functor
# -------------------------------------------------------------------
nfan() = 1, 1
nfan(n) = 1, n
nfan(out, in) = in, out
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])  # Conv Kernel
glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 5f-1) .* sqrt(24f0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(Random.GLOBAL_RNG, dims...)
# -------------------------------------------------------------------
"""
less alloc and slightly faster than Flux.Dense for CuArrays.
"""
struct Densegpu{A}
    W::CuArray{Float32, 2}
    b::CuArray{Float32, 1}
    σ::A
end
@functor Densegpu

function Densegpu(In, out, σ=identity)
    Densegpu(glorot_uniform(out, In) |> CuArray, glorot_uniform(out) |> CuArray, σ)
end

(a::Densegpu)(x::CuArray{Float32,N}) where N = a.σ.(a.W * x .+ a.b)
(a::Densegpu{identity})(x::CuArray{Float32,N}) where N = a.W * x .+ a.b

# -------------------------------------------------------------------
"""
                            -------<------<-----
                          /                      ↖
                         ↓                        |
                        \\ /                       |
               ---------- --------------          |
    In        |          |              |         |
    In      \\ |          |         ƒ    | /      ↗
    ------>     ---->----∘---->----:----    -->--
            / |        comb             | \\    state (output)
              |                         |
               -------------------------

"""
struct Recurrent{T,M,N}
    comb::T
    state::M
    ƒ::N
end
@functor Recurrent

function Recurrent(comb, ƒ, size, gpu=false)
    gpu ?
        Recurrent(comb, randn(Float32, size...) |> CuArray, ƒ) :
        Recurrent(comb, randn(Float32, size...), ƒ)
end

function (a::Recurrent)(x::T) where T
    m = a.ƒ(a.comb(a.state, x))
    a.state .= m
    return m
end
# -------------------------------------------------------------------

struct RecurrentDense{T,M,N,P,A}
    comb::T
    state::M
    W::N
    b::P
    σ::A
end
@functor RecurrentDense

function RecurrentDense(In::Int, out::Int, comb, σ=identity, gpu=false)
    gpu ?
        RecurrentDense(comb, zeros(Float32, out) |> CuArray, glorot_uniform(out, In) |> CuArray, glorot_uniform(out) |> CuArray, σ) :
        RecurrentDense(comb, zeros(Float32, out), glorot_uniform(out, In), glorot_uniform(out), σ)
end

function (a::RecurrentDense)(x::T) where T
    X = a.comb(a.state, x)
    m = a.σ.(a.W * X .+ a.b)
    a.state .= m
    return m
end

function (a::RecurrentDense{T,M,N,P,identity})(x::S) where {T,M,N,P,S}
    X = a.comb(a.state, x)
    m = a.W * X .+ a.b
    a.state .= m
    return m
end
