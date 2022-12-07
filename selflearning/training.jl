using Flux: Params
using Flux.Optimise: AbstractOptimiser
using Flux.Zygote: withgradient

"""
function train!(loss, ps::Params, opt::AbstractOptimiser, minibatches, fvals, max_steps; ϵ = 0.5, max_loss = 0.0, opt_type = :worstcase, kwargs...)
function train!(loss, ps::Params, opt::AbstractOptimiser, minibatches, fvals, mbsampler, max_steps, max_loss)

optimizes the `loss` with respect to parameters `ps` using optimiser `opt`

`minibatches` --- `AbstractVector` of minibatches
`fvals` --- container for values of the loss function on each minibatch
`mbsampler` --- function without parameters returning index of one minibatch 
`max_steps` --- maximum number of steps
`max_loss` --- stops of all losses are smaller
opt_type --- type of optimization: either `:mean` error or `:worst`-case error
`ϵ` --- exploration in worst-case optimization
"""
function train!(loss, ps::Params, opt::AbstractOptimiser, minibatches, fvals, steps; ϵ = 0.5, max_loss = 0.0, opt_type = :worst, kwargs...)
	if opt_type == :worst
		worst_sampler() = sample_minibatch(fvals, ϵ)
		train!(loss, ps, opt, minibatches, fvals, worst_sampler, steps, max_loss)
	elseif opt_type == :mean
		mean_sampler() = rand(1:length(minibatches))
		train!(loss, ps, opt, minibatches, fvals, mean_sampler, steps, max_loss)
	else
		error("unknown type of optimization $(opt_type), supported is `:worst` and `:mean`")
	end
end

function train!(loss, ps::Params, opt::AbstractOptimiser, minibatches, fvals, mbsampler, max_steps, max_loss)
	for _ in 1:max_steps
		j = mbsampler()
		d = minibatches[j]
		l, gs = withgradient(() -> loss(d), ps)
		!isfinite(l) && error("Loss is $l on data item $j")
		fvals[j] = l
		Flux.Optimise.update!(opt, ps, gs)
		all(l ≤ max_loss for l in fvals) && break
	end
	fvals
end

"""
	sample_minibatch(w, ϵ)

	samples minibatch with respect to weight w. It checks first that 
	all minibatches are finite (loss has been calculated and then it assigns 
	a valid value)
"""
function sample_minibatch(w, ϵ)
	rand() ≤ ϵ && return(rand(1:length(w))) 
	T = eltype(w)
	i = findfirst(==(typemax(T)), w)
	(i !== nothing) ? i : sample(StatsBase.Weights(w))
end