module diffutils
    export Split, Join, Parallelism
    export Nonlinear
    export WCell

    export Densegpu	

    include("./arch.jl")
    include("./layers.jl")
end
