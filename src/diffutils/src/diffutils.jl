module diffutils
    
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
    
    export Split, Join, Parallelism
    export Nonlinear, NonlinearD
    export WCell
    
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
    
    function (m::Parallelism)(x::T) where T
        m.combine(map(f -> f(x), m.layers)...)
    end
    # -------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------- 
    """
                        . Dense             . batchnorm
                        .                   .
                        .                   .
                        .                   .
                        .                   .
                        .        (id)       .      (ReLu)
        ----->          .     -------->     .     -------->
                        .                   .
                        .                   .
                        .                   .
                        .                   .
                        .                   .
                  In -> .        out        . -> out
    """
    struct Nonlinear{D <: Dense, E <: BatchNorm}
        In::D
        norm::E
    end
    
    function Nonlinear(In::Int, out::Int = In, σ = relu)
        Nonlinear(
            Dense(In, out),
            BatchNorm(out, σ))
    end
        
    @functor Nonlinear
    
    function (m::Nonlinear)(x::T) where T
        x |> m.In |> m.norm
    end
    # -------------------------------------------------------------------
    """
                   . Dense             . batchnorm         . Dense
                   .                   .                   .
                   .                   .                   .
                   .                   .                   .
                   .                   .                   .
                   .        (id)       .      (ReLu)       .       (id)
        ----->     .     -------->     .     -------->     .     ------->
                   .                   .                   .
                   .                   .                   .
                   .                   .                   .
                   .                   .                   .
                   .                   .                   .
             In -> .        out        .       hidd        . -> out
    """
    struct NonlinearD{D <: Dense, E <: BatchNorm, F <: Dense}
        In::D
        norm::E
        out::F
    end
    
    function NonlinearD(In::Int, hidd::Int = In, out::Int = In, σ = relu)
        NonlinearD(
            Dense(In, hidd),
            BatchNorm(hidd, σ),
            Dense(hidd, out))
    end
        
    @functor NonlinearD
    
    function (m::NonlinearD)(x::T) where T
        x |> x -> flatten(x) |> m.In |> m.norm |> m.out
    end
    # -------------------------------------------------------------------
    """
                      ----------:-----------(σ)-----
                    /                                \\
                   /                                  \\
        ------------------------:-------------->  W .* η ------->  
    
        .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                           η                           ∣η∣ = 1    
                           ---------:------:-----(σ)-------    
        In               /                                  \\
        ----------------<                                    >- W .* η -->
                         \\                                  /
                           ---------:------:------:---------
                           W                      ∣W∣ = ∣in∣
    """
    struct WCell{K, J}
        weightlayer::K
        layer::J
    end
    @functor WCell
    
    function WCell(weigthlayer, layer)
        return Parallelism(.*,
        Chain(weigthlayer, FullyConnected(In, In ÷ 2, 1, σ)),
        layer)
    end
    
    function (m::WCell)(x::T) where T
        x |> m.weightlayer |> m.layer
    end

end
