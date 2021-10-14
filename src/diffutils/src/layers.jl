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

    function Densegpu(in, out, σ=identity)
        Densegpu(glorot_uniform(out, in) |> CuArray, glorot_uniform(out) |> CuArray, σ)
    end
    @functor Densegpu

    (a::Densegpu)(x::CuArray{Float32,N}) where {Float32,N} = a.σ.(a.W * x .+ a.b)
    (a::Densegpu{identity})(x::CuArray{Float32,N}) where {Float32,N} = a.σ(a.W * x .+ a.b)
    # -------------------------------------------------------------------
