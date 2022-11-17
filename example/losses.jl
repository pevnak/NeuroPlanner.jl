l₂loss(model, x, y) = Flux.Losses.mse(vec(model(x)), y)
l₂loss(model, xy::NamedTuple) = l₂loss(model, xy.x, xy.y)

"""
lₛloss(x, g, H₊, H₋)

Minimizes `L*` loss, We want ``f * H₋ .< f * H₊``, which means to minimize cases when ``f * H₋ .> f * H₊``
"""
function lₛloss(model, x, g, H₊, H₋)
	g = reshape(g, 1, :)
	f = model(x) .+ g
	mean(softplus.(f * H₋ .- f * H₊))
end

lₛloss(model, xy::NamedTuple) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)

nonempty(s::NamedTuple{(:x, :sol_length, :H₊, :H₋, :path_cost, :stats)}) = !isempty(s.H₊) && !isempty(s.H₋)
nonempty(s::NamedTuple{(:x, :y, :stats)})  = true

function get_loss(name)
	name == "l2" && return((l₂loss, prepare_minibatch_l2))
	name == "l₂" && return((l₂loss, prepare_minibatch_l2))
	name == "lstar" && return((lₛloss, prepare_minibatch_lₛ))
	name == "lₛ" && return((lₛloss, prepare_minibatch_lₛ))
	error("unknown loss $(name)")
end