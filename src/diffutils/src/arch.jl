
using Flux: @functor

# ------------------------------------------------------------------- 
# Control Flow                                                                                     
# ------------------------------------------------------------------- 
"""
             .    
             .            .............................
             .            :  ->  :  ->  :  ->  :  ->  :   (----->
             .   ----->   .............................
             . 
             .            .............................
    ----->   .   ----->   :  ->  :  ->  :  ->  :  ->  :    ----->
             .            .............................
             .                                          
             .   ----->   .............................
             .            :  ->  :  ->  :  ->  :  ->  :    ----->)
             .            .............................
             .         
"""                                                      
struct Split{T}
    layers::T
end
@functor Split

Split(layers...) = Split(layers)
Split(n::Int, layer) = Split((layer for _ ∈ 1:n)...)

function (m::Split)(x::T) where T
    tuple(map(f -> f(x), m.layers))
end
# -------------------------------------------------------------------
"""
                                                       . combine
              .............................            .
    (----->   .  ->  :  ->  :  ->  :  ->  .            . 
              .............................   ----->   . 
                                                       .
              .............................            .
     ----->   .  ->  :  ->  :  ->  :  ->  .   ----->   .    ----->
              .............................            .
                                                       .
              .............................   ----->   .
     ----->)  .  ->  :  ->  :  ->  :  ->  .            .
              .............................            .
                                                       .
"""
struct Join{T, F}
    combine::F
    layers::T
end
@functor Join

Join(combine, layers...) = Join(combine, layers)
Join(combine, n::Int, layer) = Join(combine, (layer for _ ∈ 1:n)...)

function (m::Join)(xs::NTuple{N, T}) where {N, T}
    m.combine(map((f, x) -> f(x), m.layers, xs)...)
end
# -------------------------------------------------------------------
"""
             .                                      . combine
             .    ----:----:----:----:----:--->     . 
             .                                      .
             .    ----:----:----:----:----:--->     .
             .                                      .
    ----->   .    ----:----:----:----:----:--->     .    ----->
             .                                      .
             .    ----:----:----:----:----:--->     .
             .                                      .
             .    ----:----:----:----:----:--->     .
             .                                      .
"""
struct Parallelism{U,C}
    combine::U
    layers::C
end
@functor Parallelism

Parallelism(combine, layers...) = Parallelism(combine, layers)
Parallelism(combine, n::Int, layer) = Parallelism(combine, (layer for _ ∈ 1:n)...)

function (m::Parallelism)(x::T) where T
    m.combine(map(f -> f(x), m.layers)...)
end
# -------------------------------------------------------------------
"""
                            -------<------<-----
                          /                      ↖
                         |                        |
                        \\↓/                       |
               ---------- --------------          |
              |          |              |         |
    In      \\ |          |         ƒ    | /      ↗
    ----->-- → ----->----o---->----:---- → -->--
            / |        comb             | \\    state (output)
              |                         |
               -------------------------

"""
struct Recurrent{T,M,N}
    comb::T
    ƒ::N
    state::M
end
@functor Recurrent

function Recurrent(comb, ƒ, size::Int, gpu=false)
    gpu ?
        Recurrent(comb, ƒ, randn(Float32, size...) |> CuArray) :
        Recurrent(comb, ƒ, randn(Float32, size...))
end

function (a::Recurrent)(x::T) where T
    m = a.ƒ(a.comb(a.state, x))
    a.state .= m
    return m
end
# -------------------------------------------------------------------
"""
                  --------:---------(σ)---------
                /                                \\
               /                                  \\
    -------------------:------------------->  W .* η ------->  

    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                       η                            ∣η∣ = 1    
                       ------:----------------(σ)------    
    In               /                                  \\
    ----------------<                                    >- W .* η -->
                     \\                                  /
                       ------:-------------------------
                       W
"""
struct WCell{K, J}
    W::K
    ƒ::J
end
@functor WCell

function (m::WCell)(x::T) where T
    m.ƒ(x) .* σ.(m.W(x))
end
# -------------------------------------------------------------------
"""
                  --------:---------(σ)---------
                /        /                       \\
               /        /                         \\
    -------------------:------------------->  W .* η ------->  

    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                       η                            ∣η∣ = 1    
                       ----------:--------------(σ)----    
    In               /          /                       \\
    ----------------<          /                         >- W .* η -->
                     \\        /                         /
                       ------:-------------------------
                       W
"""
struct CWCell{K,J,L}
    W::K
    ƒ::J
    comb::L
end
@functor CWCell

CWCell(W, ƒ) = CWCell(W, ƒ, (x, y) -> x .* y)

function (m::CWCell)(x::T) where T
    l = m.ƒ(x)
    return l .* σ.(m.W(m.comb(x, l)))
end
# -------------------------------------------------------------------
