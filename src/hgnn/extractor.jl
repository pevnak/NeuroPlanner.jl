struct HGNN{DO,TO,MP<:NamedTuple, P<:Union{Dict,Nothing},KB<:Union{Nothing, KnowledgeBase},S<:Union{Nothing,Matrix},G<:Union{Nothing,Matrix}}
	domain::DO
	type2obs::TO
	model_params::MP
	predicate2id::P
	kb::KB
	start::S
	goal::G

	function HGNN(domain::DO, type2obs::TO, model_params::MP, predicate2id::P, kb::KB, start::S, goal::G) where {DO,TO,MP<:NamedTuple,P,KB,S,G}
		@assert issubset((:message_passes, :residual, :lite), keys(model_params)) "Parameters of the model are not fully specified"
		@assert (start === nothing || goal === nothing)  "Fixing start and goal state is bizzaare, as the extractor would always create a constant"
		new{DO,TO,MP,P,KB,S,G}(domain, type2obs, model_params, predicate2id, kb, start, goal)
	end
end

isspecialized(ex::HGNN) = (ex.predicate2id !== nothing) && (ex.kb !== nothing)
hasgoal(ex::HGNN) = ex.start !== nothing || ex.goal !== nothing

function HGNNLite(domain; message_passes = 2, residual = :linear)
	model_params = (;message_passes, residual, lite = true)
	HGNN(domain, nothing, model_params, nothing, nothing, nothing, nothing)
end

function HGNN(domain; message_passes = 2, residual = :linear, lite = true)
	model_params = (;message_passes, residual, lite = false)
	HGNN(domain, nothing, model_params, nothing, nothing, nothing, nothing)
end

function Base.show(io::IO, ex::HGNN)
	s = isspecialized(ex) ? "Specialized" : "Unspecialized"
	s *=" HGNN for $(ex.domain.name )"
	if isspecialized(ex)
		s *= " ($(length(ex.predicate2id)))"
	end
	print(io, s)
end

function specialize(ex::HGNN, problem)
	# map containing lists of objects of the same type
	type2obs = type2objects(ex.domain, problem)

	# create a map mapping predicates to their ID, which is pr
	predicates = mapreduce(union, values(ex.domain.predicates)) do p 
		allgrounding(problem, p, type2obs)
	end
	predicate2id = Dict(reverse.(enumerate(predicates)))
	ex = HGNN(ex.domain, type2obs, ex.model_params, predicate2id, nothing, nothing, nothing)

	# just add fake input for a se to have something, it does not really matter what
	n = length(predicate2id)
	x = zeros(Float32, 0, n)
	kb = KnowledgeBase((;pred_1 = x))
	for i in 1:ex.model_params.message_passes
		input_to_gnn = last(keys(kb))
		kb = encode_actions(ex, kb,  input_to_gnn)
		if ex.model_params.residual !== :none #if there is a residual connection, add it 
			res_layers = filter(s -> startswith("$s","pred_"), keys(kb))
			if length(res_layers) ≥ 2
				kb = add_residual_layer(kb, res_layers[end-1:end], n, "pred")
			end
		end
	end
	s = last(keys(kb))
	kb = append(kb, :o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))

	HGNN(ex.domain, type2obs, ex.model_params, predicate2id, kb, nothing, nothing)
end

function add_goalstate(ex::HGNN, problem, goal = goalstate(ex.domain, problem))
	ex = isspecialized(ex) ? ex : specialize(ex, problem) 
	x = encode_input(ex, goal)
	HGNN(ex.domain, ex.type2obs, ex.model_params, ex.predicate2id, ex.kb, nothing, x)
end

function add_startstate(ex::HGNN, problem, start = initstate(ex.domain, problem))
	ex = isspecialized(ex) ? ex : specialize(ex, problem) 
	x = encode_input(ex, start)
	HGNN(ex.domain, ex.type2obs, ex.model_params, ex.predicate2id, ex.kb, x, nothing)
end

function (ex::HGNN)(state)
	x = encode_input(ex::HGNN, state)
	if ex.goal !== nothing
		x = vcat(x, ex.goal)
	end 

	if ex.start !== nothing
		x = vcat(ex.start, x)
	end 
	kb = ex.kb
	kb = @set kb.kb.pred_1 = x 
	kb
end

function encode_input(ex::HGNN, state)
	@assert isspecialized(ex) "Extractor is not specialized for a problem instance"
	x = zeros(Float32, 1, length(ex.predicate2id))
	for p in PDDL.get_facts(state)
		x[ex.predicate2id[p]] = 1
	end
	x
end

function encode_actions(ex::HGNN, kb::KnowledgeBase, kid::Symbol)
	actions = ex.domain.actions
	ns = tuple(keys(actions)...)
	xs = map(values(ex.domain.actions)) do action
		xa, kb = encode_action(ex, kb, action, kid)
		xa
	end
	o = length(xs) == 1 ? only(xs) : ProductNode(NamedTuple{ns}(tuple(xs...)))
	ln = layer_name(kb, "pred")
	append(kb, ln, o)
end

function encode_action(ex::HGNN, kb::KnowledgeBase, action::GenericAction, kid::Symbol)
	preds = allgrounding(action, extract_predicates(action.precond), extract_predicates(action.effect), ex.type2obs)
	#encode edge 
	l = length(first(preds).senders)
	xs = map(1:l) do i 
		syms = [p.senders[i] for p in preds]
		ArrayNode(KBEntry(kid, [ex.predicate2id[s] for s in syms]))
	end 
	x_edge = length(xs) == 1 ? only(xs) : ProductNode(tuple(xs...))

	# check if the edge with same name has been define and add it
	previous_edges = filter(s -> startswith(String(s), "$(action.name)_"), keys(kb))
	if !ex.model_params.lite && !isempty(previous_edges)
		previous_edge = last(previous_edges)
		x_edge = ProductNode((x_edge, ArrayNode(KBEntry(previous_edge, 1:nobs(x_edge)))))
	end
	ln = layer_name(kb, "$(action.name)")
	kb = append(kb, ln, x_edge)

	# reduce to corresponding predicates
	bags = [Int[] for _ in 1:length(ex.predicate2id)]
	for (j, ps) in enumerate(preds)
		for a in ps.receivers
			a ∉ keys(ex.predicate2id) && continue
			push!(bags[ex.predicate2id[a]], j)
		end
	end
	xa = BagNode(ArrayNode(KBEntry(ln, 1:nobs(x_edge))), ScatteredBags(bags))
	xa, kb
end


function allgrounding(action::GenericAction, senders::Vector{<:Term}, receivers, type2obs::Dict; unique_args = true)
	types = [type2obs[k] for k in action.types]
	assignments = vec(collect(Iterators.product(types...)))
	unique_args && filter!(v -> length(unique(v)) == length(v), assignments) 
	as = map(assignments) do v 
		assignment = Dict(zip(action.args, v))
		(senders = [ground(p, assignment) for p in senders], 
		 receivers = [ground(p, assignment) for p in receivers])
	end 
end

