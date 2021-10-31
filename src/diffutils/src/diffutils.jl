module diffutils
    export Split, Join, Parallelism
    export Nonlinear
    export WCell, CWCell

    export Densegpu	

    include("./arch.jl")
    include("./layers.jl")
end
