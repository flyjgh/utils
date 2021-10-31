using Flux: @functor
using Flux: Chain
using Flux: Dense
using Flux: BatchNorm
using Flux: flatten
using Flux: Recur
using Flux: LSTM
using Flux: relu
using Flux: σ
using Base: @pure

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
struct Nonlinear{D <: Dense, E <: BatchNorm}
    In::D
    norm::E
end
@functor Nonlinear

function Nonlinear(In::Int, out::Int=In, σ=relu)
    Nonlinear(
        Dense(In, out),
        BatchNorm(out, σ))
end
    
function (m::Nonlinear)(x::T) where T
    x |> m.In |> m.norm
end
# -------------------------------------------------------------------
"""
                  --------:---------(σ)---------
                /                                \\
               /                                  \\
    -------------------:------------------->  W .* η ------->  

    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                       η                            ∣η∣ = 1    
                       ------:----(ReLu)------(σ)------    
    In               /                                  \\
    ----------------<                                    >- W .* η -->
                     \\                                  /
                       ------:----(ReLu)---------------
                       W                        ∣W∣ = out
"""
struct WCell{K, J}
    weight::K
    layer::J
end
@functor WCell

function (m::WCell)(x::T) where T
    m.layer(x) .* σ.(m.weight(x))
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
                       W                        ∣W∣ = out
"""
struct CWCell{K,J,L<:Nonlinear}
    weight::K
    layer::J
end
@functor CWCell

function CWCell(weigth, layer, In::Int, layer_out::Int)
    println("Weight layer input must be of size (In + layer_out)")
    println("It must output a scalar")
    return CWCell(weigth, layer, Nonlinear(In + layer_out, 1))
end

function (m::CWCell)(x::T) where T
    l = m.layer(x)
    return l .* σ(m.weight(vcat(x, l)))
end
# -------------------------------------------------------------------
