using Flux
using Plots
using Images
using Metalhead
using Flux: @epochs
using BenchmarkTools
using Metalhead.Images
include("C:/Users/jgh/prog/utils/src/diffutils/src/diffutils.jl")
using .diffutils

x=load("C:/Users/jgh/Documents/J-art/animalfaces/pixabay_wild_000129.jpg")

function load_img(dir)
    load.(readdir(dir, join=true))
end

function prepare_data(dir, batchsize)
    x = load_img(dir)
    x = permutedims(channelview(x), (2,3,1))
    x = cat(x, reverse(x, dims=1), dims=4)
    X = DataLoader(x,batchsize=batchsize,shuffle=true,partial=false) |> gpu
    return X, x
end

X = prepare_data("C:/Users/jgh/Documents/J-art/animalfaces", 32)

l = ResNet().layers

function gen()
    Chain(
        l[1:end-3],
        Dense(2048, 4096),
        BatchNorm(4096),
        x -> x .|> leakyrelu,
        Dense(4096, 4096),
        Dense(4096, 512 * 512 * 3, σ),
        x -> reshape(x, 512, 512, 3)) |> gpu
end

function disc()
    Chain(
         l[1:end-3],
         Dense(2048, 256, leakyrelu),
         BatchNorm(256),
         x -> x .|> leakyrelu,
         Dense(256, 256),
         Dense(256, 1, σ)) |> gpu
end

@inline function noise()
    randn(Float32, 512, 512, 3) .|> gpu
end

@inline KLDivergence(ŷ, y) = sum(ŷ .* log.(ŷ ./ y)) |> gpu

@inline function G_loss(x, g, d, 𝒩)
    generated = g(𝒩)
    dfk = d(generated) |> gpu
    BCE = bce(dfk, one(Float32)) |> gpu
    KLD = 0.5f0 * sum(@. (exp.(2f0 .* generated) .+ x .^ 2 .-1f0 .- 2f0 .* generated)) / len
    return BCE + KLD |> gpu
end

@inline function D_loss(x, g, d, ℛ, 𝓕, 𝒩)
    dfk = d(g(𝒩)) |> gpu
    drl = d(x) |> gpu
    BCE = bce(drl, ℛ) + bce(dfk, 𝓕) |> gpu
    return BCE |> gpu
end

@inline function train_step(x, g, d, ℛ, 𝓕, opt, 𝒩, 𝒢params, 𝒟params)
    ∂gen  = ∂(() -> G_loss(x, g, d, 𝒩), 𝒢params) |> gpu
    ∂disc = ∂(() -> D_loss(x, g, d, ℛ, 𝓕, 𝒩), 𝒟params) |> gpu
    update!(opt, 𝒢params, ∂gen)  |> gpu
    update!(opt, 𝒟params, ∂disc) |> gpu
end

function train()
    X = load_img("C:/Users/jgh/prog/maths/ml/gan/data/animalfaces")
    g = gen()  |> gpu
    d = disc() |> gpu
    ℛ = ones(Float32, 1, 32) |> gpu
    𝓕 = zeros(Float32, 1, 32) |> gpu
    𝒢params = params(g)
    𝒟params = params(d)
    opt = ADAM(3f-5 , (0.9f0, 0.999f0))
    "☼-----begin >>"
    @epochs 10 for data ∈ X
            𝒩 = noise() |> gpu
            train_step(X, g, d, ℛ, 𝓕, opt, 𝒩, 𝒢params, 𝒟params)
    end
end

train()


# X = Float32 - (512, 512, 3)
# g = gen()
# d = disc()
# ℛ = ones(Float32, 1, 32) |> gpu
# 𝓕 = zeros(Float32, 1, 32) |> gpu
# opt = ADAM(3f-5 , (0.9f0, 0.999f0))
