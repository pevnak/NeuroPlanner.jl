struct HGNNLite{DO,TO,P<:Union{Dict,Nothing},KB<:Union{Nothing, KnowledgeBase},G<:Union{Nothing,Matrix}}
	domain::DO
	type2obs::TO
	model_params::NamedTuple{(:message_passes, :residual), Tuple{Int64, Symbol}}
	predicate2id::P
	kb::KB
	goal::G
end

isspecialized(ex::HGNNLite) = (ex.predicate2id !== nothing) && (ex.kb !== nothing)
hasgoal(ex::HGNNLite) = ex.goal !== nothing

function HGNNLite(domain; message_passes = 2, residual = :linear)
	model_params = (;message_passes, residual)
	HGNNLite(domain, nothing, model_params, nothing, nothing, nothing)
end

function Base.show(io::IO, ex::HGNNLite)
	s = isspecialized(ex) ? "Specialized" : "Unspecialized"
	s *=" HGNNLite for $(ex.domain.name )"
	if isspecialized(ex)
		s *= " ($(length(ex.predicate2id)))"
	end
	print(io, s)
end

function specialize(ex::HGNNLite, problem)
	# map containing lists of objects of the same type
	type2obs = map(unique(values(problem.objtypes))) do k 
		k => [n for (n,v) in problem.objtypes if v == k]
	end |> Dict

	# create a map mapping predicates to their ID, which is pr
	predicates = mapreduce(union, values(ex.domain.predicates)) do p 
		allgrounding(problem, p, type2obs)
	end
	predicate2id = Dict(reverse.(enumerate(predicates)))
	ex = HGNNLite(ex.domain, type2obs, ex.model_params, predicate2id, nothing, nothing)

	# just add fake input for a se to have something, it does not really matter what
	n = length(predicate2id)
	x = zeros(Float32, 0, n)
	kb = KnowledgeBase((;x1 = x))
	sₓ = :x1
	for i in 1:ex.model_params.message_passes
		input_to_gnn = last(keys(kb))
		ds = encode_actions(ex, input_to_gnn)
		kb = append(kb, layer_name(kb, "gnn"), ds)
		if ex.model_params.residual !== :none #if there is a residual connection, add it 
			kb = add_residual_layer(kb, keys(kb)[end-1:end], n)
		end
	end
	s = last(keys(kb))
	kb = append(kb, :o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))

	HGNNLite(ex.domain, type2obs, ex.model_params, predicate2id, kb, nothing)
end

function add_goalstate(ex::HGNNLite, problem, goal = goalstate(ex.domain, problem))
	ex = isspecialized(ex) ? ex : specialize(ex, problem) 
	x = encode_input(ex, goal)
	HGNNLite(ex.domain, ex.type2obs, ex.model_params, ex.predicate2id, ex.kb, x)
end

function (ex::HGNNLite)(state)
	x = encode_input(ex::HGNNLite, state)
	if hasgoal(ex)
		x = vcat(x, ex.goal)
	end 
	kb = ex.kb
	kb = @set kb.kb.x1 = x 
	kb
end

function encode_input(ex::HGNNLite, state)
	@assert isspecialized(ex) "Extractor is not specialized for a problem instance"
	x = zeros(Float32, 1, length(ex.predicate2id))
	for p in PDDL.get_facts(state)
		x[ex.predicate2id[p]] = 1
	end
	x
end

function encode_actions(ex::HGNNLite, kid::Symbol)
	actions = ex.domain.actions
	ns = tuple(keys(actions)...)
	xs = map(values(ex.domain.actions)) do action
		encode_action(ex, action, kid)
	end
	length(xs) == 1 ? only(xs) : ProductNode(NamedTuple{ns}(tuple(xs...)))
end

function encode_action(ex::HGNNLite, action::GenericAction, kid::Symbol)
	preds = allgrounding(action, ex.type2obs)
	encode_predicates(ex, preds, kid)
end

function encode_predicates(ex::HGNNLite, preds, kid::Symbol)
	l = length(first(preds))
	xs = map(1:l) do i 
		syms = [p[i] for p in preds]
		ArrayNode(KBEntry(kid, [ex.predicate2id[s] for s in syms]))
	end 

	bags = [Int[] for _ in 1:length(ex.predicate2id)]
	for (j, ps) in enumerate(preds)
		for a in ps
			a ∉ keys(ex.predicate2id) && continue
			push!(bags[ex.predicate2id[a]], j)
		end
	end
	BagNode(ProductNode(tuple(xs...)), ScatteredBags(bags))
end
