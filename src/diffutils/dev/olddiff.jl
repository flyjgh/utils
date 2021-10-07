module diffutils
    
    using Flux: @functor
    using Flux: Chain
    using Flux: Dense
    using Flux: BatchNorm
    using Flux: Recur
    using Flux: LSTM
    using Flux: relu
    using Flux: σ
    using Base: @pure
    
    export Split, Join, Parallelism, IBO
    export FullyConnected, Nonlinear
    export LSTMBlock, RngLSTM
    export CWNN, CWCell
    export CWDNL, CWLSTM
    
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
        branches::T
    end
    @functor Split
    
    Split(branches...) = Split(branches)
    
    @pure function (m::Split)(x::Array{T,N}) where {T, N}
        tuple(map(f -> f(x), m.branches))
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
        branches::T
    end
    @functor Join
    
    Join(combine, branches...) = Join(combine, branches)
    
    @pure function (m::Join)(xs::NTuple{N, Array{T,M}}) where {T, N, M}
        m.combine(map((f, x) -> f(x), m.branches, xs)...)
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
        branches::C
    end
    @functor Parallelism
    
    Parallelism(combine, branches::Tuple) = Parallelism(combine, branches)
    Parallelism(combine, branches...) = Parallelism(combine, branches)
    Parallelism(branches::Tuple) = Parallelism(vcat, branches)
    
    
    @pure function (m::Parallelism)(x::Array{T,N}) where {T, N}
        m.combine(map(f -> f(x), m.branches)...)
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
    struct Nonlinear{D <: Dense, E <: BatchNorm, F <: Dense}
        In::D
        norm::E
        out::F
    end
    
    function Nonlinear(In::Int, hidd::Int = In, out::Int = In, σ = relu)
        Nonlinear(
            Dense(In, hidd),
            BatchNorm(hidd, σ),
            Dense(hidd, out))
    end
        
    @functor Nonlinear
    
    @pure function (m::Nonlinear)(x::Vector{T}) where T
        x |> m.In |> m.norm |> m.out
    end
    
    @pure function (m::Nonlinear)(x::Array{T,N}) where {T, N}
        x |> x -> flatten(x) |> m.In |> m.norm |> m.out
    end
    # -------------------------------------------------------------------
    """
                      ----:----(ReLu)------:------(σ)------
                    /                                       \\
                   /                                         \\
        -----:-----:---------:---------(ReLu)--------->  W .* η ------->  
    
        .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                             η                                     ∣η∣ = 1    
                            ------:----(ReLu)------:-----:-----(σ)-----                
        In                /                                             \\
        -----:-----:-----<                                               >- W .* η -->
                          \\                                             /
                            -----:-----(ReLu)-----:--------------------     
                            W                            ∣W∣ = ∣in∣
    """
    struct CWCell{K, J}
        ff::K
        parallel::J
    end
    
    function CWDff(n::Int, In::Int)
        return (Nonlinear(In) for _ ∈ 1:n)
    end
    
    function CWDSplit(In::Int)
        return Parallelism(.*,
        Chain(Nonlinear(In)..., FullyConnected(In, In ÷ 2, 1, σ)),
        Nonlinear(In))
    end
    
    CWCell(n::Int, In::Int) = CWCell(CWDff(n, In)..., CWDSplit(In))
    CWCell(args...) = return CWCell(args[begin:end-1], args[end])
    
    @functor CWCell
    
    @pure function (m::Nonlinear)(x::Array{T,N}) where {T, N}
        x |> m.ff |> m.parallel
    end
    # -------------------------------------------------------------------
    # High level architecture                                                
    # -------------------------------------------------------------------
    """
         N:ϵ:M                                            . vcat
                  ...............                         . 
        (----->   .  <>  :  <>  .                 ---->   .
                  ...............                  ∣e∣     . 
                      1 ... N                             .
                  ......................                  .     
         ----->   .  <>  :  <>  :  <>  .          ---->   .    ----->
                  ......................           ∣e∣     .
                     1     ...    1+ϵ                     . 
                  .............................           . 
         ----->)  .  <>  :  <>  :  <>  :  <>  .   ---->   .
                  .............................    ∣e∣     . 
                     1      2     ...     M               .
    """
    struct RngLSTM{T}    
        branches::T
    end
    
    function RngLSTM(r::AbstractRange, e = 1)
        RngLSTM((LSTM(i, e) for i ∈ r)...)
    end
    
    @functor RngLSTM
                                                           
    @pure function (m::RngLSTM)(xs::NTuple{N, Array{T,M}}) where {T, N, M}
        vcat(map((f, x) -> f(x), m.branches, xs)...)
    end

end
