
using Zygote

# -----------------------------------------------------------------------------

struct Layer{N,M,A} <: Function
    W
    b
    σ::A
end

Layer(In,out=In,σ=identity) = Layer{In,out,typeof(σ)}(rand(out,In), rand(out), σ)

(a::Layer)(x) = a.σ(a.W * x .+ a.b)

struct Seq{T<:Tuple}
    layers::T
    Seq(layers) = new{typeof(reverse(layers))}(reverse(layers))
    Seq(layers...) = new{typeof(reverse(layers))}(reverse(layers))
end

(m::Seq)(x) = ∘(m.layers...)(x)

Base.getindex(m::Seq, i) = Seq(reverse(m.layers)[i])
Base.lastindex(m::Seq) = length(m.layers)
Base.show(io::IO, m::Seq) = print(io, "::Seq = " * join(typeof.(reverse(m.layers)), ", ") * ")")

struct Opt 
    η
    ρ
    v
end

Opt(η=1e-3,ρ=0.9) = Opt(η, ρ, [])

function (o::Opt)(x,∇)
    x.W .-= o.ρ .* o.v .- o.η .* ∇[:W]
    x.b .-= o.ρ .* o.v .- o.η .* ∇[:b]
    o.v .= abs.(o.ρ .* o.v .- ∇)
end

l = Layer(3)
o = Opt()
o(l, rand(3,3), rand(3))

loss(ŷ,y) = ((ŷ .- y) .^ 2) / 

function trainstep(layer, unknowns)
    δ = ((layer) -> loss(layer, rand(unknowns,100)))'(layer)
    layer.W .-= 1e-3δ[:W]
    layer.b .-= 1e-3δ[:b]
    return loss(layer,rand(3))
end

function train(unknowns)
    layer = Layer(3, 1)
    while true
        l = trainstep(layer, unknowns)
        @show l
        l <= 0.001 && break
    end
    layer
end

x = train(3)

testeq(x)
l([1,2,3])[1] ≈ eq([1,2,3])
l = Layer(3, 3)
δ = Zygote.gradient((l) -> loss(l, rand(3)), l)[1][:W]

# -----------------------------------------------------------------------------
