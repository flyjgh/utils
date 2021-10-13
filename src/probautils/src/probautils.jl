module probautils

    using Distributions, Plots

    export fit_distrib, plotproba, plotfit

    function fit_distrib(a)
        a = a |> x -> reshape(x, :)
        distribs = [Bernoulli, Beta, Binomial, Categorical, Exponential, LogNormal, Normal, Gamma, Geometric, Pareto, Poisson, Cauchy{Float32}, Uniform, Laplace, InverseGaussian, Rayleigh]
        fits = fit.(distribs, Ref(a))
        return fits[findmax(loglikelihood.(fits, Ref(a)))[2]]
    end
    
    fit_distrib(distrib, a) = fit(distrib, a)

    function plotproba(a)
        a = a |> x -> reshape(x, :)
        histogram(a, background=:grey22)
    end

    function plotfit(a)
        fitd = fit_distrib(a)
        typeofdistrib = fitd |> typeof
        μ = string(getfield(fitd, 1))
        length(μ) >= 5 ? μ=μ[1:5] : nothing
        fieldcount(Normal() |> typeof) >= 2 ?
        begin
            σ = string(getfield(fitd, 2))[1:4]
            length(σ) >= 5 ? σ=σ[1:5] : nothing
            observations = rand(fitd, 100000)
            s1 = fieldname(a, 1)
            s2 = fieldname(a, 2)
            histogram(observations, title="$typeofdistrib", xlabel = "$s1 = $σ  /  $s2 = $μ", background=:grey22)
        end :
        begin
            s1 = fieldname(a, 1)
            histogram(observations, title="$typeofdistrib", xlabel = "$s1 = $μ", background=:grey22)
        end
    end

    function plotfit(distrib, a)
        fitd = fit_distrib(distrib, a)
        typeofdistrib = fitd |> typeof
        μ = string(getfield(fitd, 1))
        length(μ) >= 5 ? μ=μ[1:5] : nothing
        fieldcount(Normal() |> typeof) >= 2 ?
        begin
            σ = string(getfield(fitd, 2))[1:4]
            length(σ) >= 5 ? σ=σ[1:5] : nothing
            observations = rand(fitd, 100000)
            s1 = fieldname(a, 1)
            s2 = fieldname(a, 2)
            histogram(observations, title="$typeofdistrib", xlabel = "$s1 = $σ  /  $s2 = $μ", background=:grey22)
        end :
        begin
            s1 = fieldname(a, 1)
            histogram(observations, title="$typeofdistrib", xlabel = "$s1 = $μ", background=:grey22)
        end
    end
    
end

# distrib = randn(1000,1000)
# plotproba(distrib)
# plotfit(distrib)
# fit_distrib(distrib)
