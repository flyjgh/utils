module diffutils
    export Split, Join, Parallelism
    export Nonlinear
    export WCell, CWCell
    export Recurrent, RecurrentDense
    export Densegpu

    include("./arch.jl")
    include("./layers.jl")
end
