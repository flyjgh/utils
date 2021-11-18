include("C:/Users/jgh/prog/utils/src/diffutils/src/diffutils.jl")
using Flux, Plots, CUDA, Zygote, Images
using Flux: DataLoader, update!, train!, @epochs
using .diffutils
using Base: @pure

x = load("C:/Users/jgh/Documents/J-art/Sanimalfaces/pixabay_wild_000129.jpg")
t(n) = rand(Float32, 512, 512, 3, n)

function load_dir(dir)
    load.(readdir(dir, join=true))
end

@inline @pure function tensor2rgb(x::Array{Float32, 4})::Matrix{RGB{N0f8}}
    w = size(x, 1)
    h = size(x, 2)
    img = RGB{N0f8} - (w, h)
    @inbounds for j ∈ 1:h
        @inbounds for i ∈ 1:w
            img[i, j] = RGB(x[i, j, 1, 1:3]...)
        end
    end
    return img
end

@inline @pure function rgb2tensor(x::Matrix{RGB{N0f8}})::Array{Float32,4}
    w = size(x, 1)
    h = size(x, 2)
    tsr = Float32 - (w, h, 3, 1)
    @inbounds for j ∈ 1:h
        @inbounds for i ∈ 1:w
            tsr[i, j, 1, 1] = x[i, j].r
            tsr[i, j, 2, 1] = x[i, j].g
            tsr[i, j, 3, 1] = x[i, j].b
        end
    end
    return tsr
end

@inline @pure function rgb2tensor(x::Vector{Matrix{RGB{N0f8}}})::Array{Float32,4}
    w = size(x[1], 1)
    h = size(x[1], 2)
    l = length(x)
    tsr = Float32 - (w, h, 3, l)
    @inbounds for n ∈ 1:l
        @inbounds for j ∈ 1:h
            @inbounds for i ∈ 1:w
                tsr[i, j, 1, n] = x[n][i, j].r
                tsr[i, j, 2, n] = x[n][i, j].g
                tsr[i, j, 3, n] = x[n][i, j].b
            end
        end
    end
    return tsr
end


# function tobatch(x, batchsize)
#     X = []
#     l = size(x, 4)
#     nbatch = (l ÷ batchsize) - 1
#     for i ∈ 1:nbatch
#         x[:, :, :, batchsize*i:batchsize*i+batchsize] ⇒ X
#     end
#     return X 
# end

function prepare_data(dir, batchsize)
    x = load_dir(dir)
    x = rgb2tensor(x)
    x = cat(x, reverse(x, dims=1), dims=4)
    X = DataLoader(x, batchsize=batchsize, shuffle=true, partial=false)
    return X
end

# -------------------------------------------------------------------

function chainconv()
    return Chain(
        Conv((8,8), 3 => 32, pad=(4,4), stride=(8,8)),
        Dropout(0.01),
        Conv((4,4), 32 => 64, pad=(4,4), stride=(4,4)),
        Conv((4,4), 64 => 128, pad=(4,4), stride=(4,4)),
        x -> flatten(x),
        Nonlinear(4608, 2048)) |> gpu
end

function chaindeconv()
    return Chain(
        x -> reshape(x, 4, 4, 128, :),
        ConvTranspose((8,8), 128 => 64, pad=(2,2), stride=(4,4)),
        ConvTranspose((8,8), 64 => 32, pad=(2,2), stride=(4,4)),
        ConvTranspose((8,8), 32 => 16, pad=(2,2), stride=(4,4)),
        ConvTranspose((8,8), 16 => 3, pad=(3,3), stride=(2,2)),
        x -> σ.(x)) |> gpu
    end
    
autoencoder() = chaindeconv() ∘ chainconv() |> gpu

# -------------------------------------------------------------------

@pure KLD(ŷ, y, ϵ=1f-5)::Float32 = sum(ŷ .* log.((ŷ.+ϵ) ./ (y.+ϵ))) |> gpu

@pure function KLDCPU(ŷ, y, ϵ=1f-5)::Float32
    res = 0f0
    for l ∈ size(y, 4)
        for k ∈ size(y, 3)
            for j ∈ size(y, 2)
                for i ∈ size(y, 1)
                    res += ŷ[i,j,k,l] * log((ŷ[i,j,k,l]+ϵ) / (y[i,j,k,l]+ϵ))
                end
            end
        end
    end
    return res
end

function loss(ae, x)::Float32
    ŷ = ae(x) |> gpu
    KLD(ŷ, x) + KLD(x, ŷ)
end

function trainstep(ae, x, opt, 𝑝)
    ∂ = gradient(() -> loss(ae, x), 𝑝)
    update!(opt, 𝑝, ∂)
end

function train(n)
    X = prepare_data("C:/Users/jgh/Documents/J-art/XSanimalfaces", 8)
    img = load("C:/Users/jgh/Documents/J-art/XSanimalfaces/pixabay_wild_000129.jpg")
    println("☼ data loaded")
    println("...")
    μ   = 3f-4
    opt = ADAM(μ)
    ref = rgb2tensor(img)
    ae  = autoencoder()
    println("☼ model loaded")
    println("...")
    𝑝   = params(ae)
    
    println("☼---->> begin")
    @epochs n begin 
        for x ∈ X
            trainstep(ae, x |> gpu, opt, 𝑝)
        end
        plot(tensor2rgb(ae(ref)), xaxis=string(loss(ae, ref |> gpu)))
    end
end


train(10)

