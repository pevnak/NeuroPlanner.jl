using Flux: Params
using Flux.Optimise: AbstractOptimiser
using Flux.Zygote: withgradient

debug_data = nothing

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
function train!(loss, model, ps::Params, opt::AbstractOptimiser, minibatches, fvals, steps; ϵ = 0.5, max_loss = 0.0, opt_type = :worst, debug = false, kwargs...)
	if opt_type == :worst
		worst_sampler() = sample_minibatch(fvals, ϵ)
		train!(loss, model, ps, opt, minibatches, fvals, worst_sampler, steps, max_loss, debug)
	elseif opt_type == :mean
		mean_sampler() = rand(1:length(minibatches))
		train!(loss, model, ps, opt, minibatches, fvals, mean_sampler, steps, max_loss, debug)
	else
		error("unknown type of optimization $(opt_type), supported is `:worst` and `:mean`")
	end
end

function train!(loss, model, ps::Params, opt::AbstractOptimiser, minibatches, fvals, mbsampler, max_steps, max_loss, debug )
	for _ in 1:max_steps
		j = mbsampler()
		d = prepare_minibatch(minibatches[j])
		l, gs = withgradient(() -> loss(model, d), ps)
		!isfinite(l) && error("Loss is $l on data item $j")
		fvals[j] = l
		if debug 
			@show l
			global debug_data
			debug_data = deepcopy((model, d))
			isinvalid(gs) && error("gradient is isinvalid")
		end
		Flux.Optimise.update!(opt, ps, gs)
		all(l ≤ max_loss for l in fvals) && break
	end
	fvals
end

function train!(loss, model, ps::Params, opt::AbstractOptimiser, prepare_minibatch, max_steps; reset_fval = 1000, verbose = true, stop_fval=typemin(Float64), logger = nothing, trn_data = [], debug = false)
	fval, n = 0.0, 0
	last_fval = nothing
	for i in 1:max_steps
		global debug_data
		d = prepare_minibatch()
		l, gs = withgradient(() -> loss(model, d), ps)
		if debug 
			#@show l
			global debug_data
			debug_data = deepcopy((model, d))
			isinvalid(gs) && error("gradient is isinvalid")
		end
		!isfinite(l) && error("Loss is $l on data item")
		fval += l
		n += 1
		# serialize("/home/tomas.pevny/tmp/debug.jls",(model, d))
		Flux.Optimise.update!(opt, ps, gs)
		if mod(i, reset_fval) == 0
			last_fval = fval / n
			verbose && println(i,": ", round(last_fval, digits = 3))
			if logger !== nothing
				log_value(logger, "fval", last_fval; step=i)
				if !isempty(trn_data)
					f = mean(loss(model, d, x -> x > 0) for d in trn_data)
					log_value(logger, "f01", f; step=i)
				end
			end
			last_fval < stop_fval && break
			fval, n = 0.0, 0
		end
	end
	n > 0 ? fval / n : last_fval
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

prepare_minibatch(d) = d
prepare_minibatch(mb::NamedTuple{(:minibatch, :stats)}) = prepare_minibatch(mb.minibatch)



isvalid(args...) = !isinvalid(args...)
isinvalid(gs::Flux.Zygote.Grads) = any(map(isinvalid, collect(values(gs.grads))))
isinvalid(x::AbstractArray) =     any(isinf.(x)) || any(isnan.(x))
isinvalid(x::Number) = isinf(x) || isnan(x)
isinvalid(x::Tuple) = any(map(isinvalid, x))
isinvalid(x::NamedTuple) = any(map(isinvalid, x))
isinvalid(x::Nothing) = false