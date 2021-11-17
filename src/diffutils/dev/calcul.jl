
using Zygote

# -----------------------------------------------------------------------------

x = 1.
δ = (x -> 2x)'(x)

# -----------------------------------------------------------------------------

struct Unknowns
    v::Vector
end

eq(x,y,z) = x^2 + 2y + z
eq(x::Unknowns) = eq(x.v...)

loss(x, y) = (eq(x) - y) ^ 2

opt(x, ∇, η=1e-3) = x.v .-= η .* ∇

function trainstep(x, y)
    ∇ = (x -> loss(x, y))'(x)[:v]
    opt(x, ∇)
end

function train(unknowns, ans, epochs)
    x = Unknowns([rand() for _ = 1:unknowns])
    for _ = 1:epochs
        trainstep(x, ans)
        @show eq(x)
    end
    x
end

x = train(3, 2f1, 5000)

eq(x)

# -----------------------------------------------------------------------------

struct Layer <: Function
    W
    b
    Layer(In, out) = new(rand(out, In), rand(out))
end

(a::Layer)(x) = a.W * x .+ a.b

eq(x,y,z) = 3x^3 + 2y^2 + 2z

loss(layer,ŷ) = +([(layer(ŷ) .- eq.(ŷ[:,i]...))[1] ^ 2 for i=1:size(ŷ,2)]...) / size(ŷ,2)

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

obj = exp ∘ log ∘ sin
obj[2]
obj[2:3]

# -----------------------------------------------------------------------------

S  = α -> β -> γ -> (α)(γ)((β)(γ))
K  = α -> β -> α
I  = (S)(K)(K)
KI = (K)(I)
Y = ƒ -> (α -> (ƒ)(α)(α))(α -> (ƒ)(α)(α))
B = γ -> g -> α -> (γ)((g)(α))
var = α -> β -> (β)(α)

struct Combinator <: Function
    W
    b
    σ
    Combinator(In, out=In; σ=identity) = new(rand(out, In), rand(out), σ)
end

(d::Combinator)(x) = y -> d.σ.(d.W * [x,y] .+ d.b)

softmax = x -> exp.(x) ./ sum(exp.(x))
# max = x -> (x .== maximum(x))
loss = m -> γ -> ρ -> m(γ)(ρ) .- KI(γ)(ρ)

KI₂ = Combinator(2,1,σ=softmax)

for i ∈1:100
    ∇ = (x -> loss(x)(1)(0)[1])'(KI₂)
    (δ -> η -> KI₂.W .-= η*δ[:W])(∇)(3e-3)
    (δ -> η -> KI₂.b .-= η*δ[:b])(∇)(3e-3)
end

loss(KI₂)(true)(false)
KI₂(true)(false)
(x -> loss(x)(1)(0)[1])'(KI₂)

# -----------------------------------------------------------------------------
