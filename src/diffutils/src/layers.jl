using Random
using CUDA: CuArray
using Flux: @functor
using Flux: Dense
using Flux: BatchNorm

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
                  . Dense          . batchnorm
                  .                .
                  .                .
                  .                .
                  .                .
                  .      (id)      .      (ReLu)
    ----->        .    ------->    .     --------> 
                  .                .
                  .                .
                  .                .
                  .                .
                  .                .
            In -> .       out      . -> out
"""
struct Nonlinear{D<:Dense, E<:BatchNorm}
    dense::D
    norm::E
end
@functor Nonlinear

function Nonlinear(In::Int, out::Int=In, σ=relu)
    Nonlinear(
        Dense(In, out),
        BatchNorm(out, σ))
end
    
function (m::Nonlinear)(x::T) where T
    x |> m.dense |> m.norm
end

# -------------------------------------------------------------------

struct Densegpu{A}
    W::CuArray{Float32, 2}
    b::CuArray{Float32, 1}
    σ::A
end
@functor Densegpu

function Densegpu(In, out=In, σ=identity)
    Densegpu(glorot_uniform(out, In) |> CuArray, glorot_uniform(out) |> CuArray, σ)
end

function Base.show(io::IO, l::Densegpu)
    ("Densegpu(", size(l.W, 2), ", ", size(l.W, 1), ", ", l.σ, ")") .|> x -> print(io, x)
  end

(a::Densegpu)(x::CuArray{Float32,N}) where N = a.σ.(a.W * x .+ a.b)
(a::Densegpu{identity})(x::CuArray{Float32,N}) where N = a.W * x .+ a.b

# -------------------------------------------------------------------
"""
                            -------<------<-----
                          /                      ↖
                         |                        |
                        \\↓/                       |
               ---------- --------------          |
              |          |              |         |
    In      \\ |          |   σ.(W*x.+b) | /      ↗
    ----->-- → ----->----o---->----:---- → -->--
            / |        comb             | \\    state (output)
              |                         |
               -------------------------
"""
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

# -------------------------------------------------------------------
