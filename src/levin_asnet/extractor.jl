struct LevinASNet{DO,TO,P<:Union{Dict,Nothing},A<:Union{Dict,Nothing},KB<:Union{Nothing, KnowledgeBase},G<:Union{Nothing,Matrix}}
	domain::DO
	type2obs::TO
	model_params::NamedTuple{(:message_passes, :residual), Tuple{Int64, Symbol}}
	predicate2id::P
	action2id::A
	kb::KB
	goal::G
end

isspecialized(ex::LevinASNet) = (ex.predicate2id !== nothing) && (ex.kb !== nothing)
hasgoal(ex::LevinASNet) = ex.goal !== nothing

function LevinASNet(domain; message_passes = 2, residual = :linear)
	model_params = (;message_passes, residual)
	LevinASNet(domain, nothing, model_params, nothing, nothing, nothing, nothing)
end

function Base.show(io::IO, ex::LevinASNet)
	s = isspecialized(ex) ? "Specialized" : "Unspecialized"
	s *=" LevinASNet for $(ex.domain.name )"
	if isspecialized(ex)
		s *= " ($(length(ex.predicate2id)))"
	end
	print(io, s)
end

function specialize(ex::LevinASNet, problem)
	# map containing lists of objects of the same type
	type2obs = type2objects(ex.domain, problem)

	# create a map mapping predicates to their ID, which is pr
	predicates = mapreduce(union, values(ex.domain.predicates)) do p 
		allgrounding(problem, p, type2obs)
	end
	predicate2id = Dict(reverse.(enumerate(predicates)))
	ex_hgnn = HGNN(ex.domain, type2obs, merge(ex.model_params, (; lite = false)), predicate2id, nothing, nothing)

	# just add fake input for a se to have something, it does not really matter what
	n = length(predicate2id)
	x = zeros(Float32, 0, n)
	kb = KnowledgeBase((;pred_1 = x))
	for i in 1:ex.model_params.message_passes
		input_to_gnn = last(keys(kb))
		kb = encode_actions(ex_hgnn, kb,  input_to_gnn)
		if ex_hgnn.model_params.residual !== :none #if there is a residual connection, add it 
			res_layers = filter(s -> startswith("$s","pred_"), keys(kb))
			if length(res_layers) ≥ 2
				kb = add_residual_layer(kb, res_layers[end-1:end], n, "pred")
			end
		end
	end
	s = last(keys(kb))
	kb = append(kb, :heuristic, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))

	ex = LevinASNet(ex.domain, type2obs, ex.model_params, predicate2id, nothing, nothing, nothing)
	ds, preds = encode_policy(ex, s)
	action2id = Dict(reverse.(enumerate(preds)))
	for k in keys(ds)
		kb = append(kb, Symbol("policy_$(k)"), ds[k])
	end
	LevinASNet(ex.domain, type2obs, ex.model_params, predicate2id, action2id, kb, nothing)
end

"""
encode_policy(ex::LevinASNet, kid::Symbol)

create an output with item per action
"""
function encode_policy(ex::LevinASNet, kid::Symbol)
	actions = ex.domain.actions
	ns = tuple(keys(actions)...)
	xs = map(values(ex.domain.actions)) do action
		ds, preds = encode_policy(ex, action, kid)
		preds = [tuple(action.name, x...) for x in preds]
		(ds, preds)
	end
	preds = reduce(vcat, [x[2] for x in xs])
	ds = NamedTuple{ns}(tuple([x[1] for x in xs]...))
	ds, preds
end

function encode_policy(ex::LevinASNet, action::GenericAction, kid::Symbol)
	preds = allgrounding(action, ex.type2obs)
	l = length(first(preds))
	xs = map(1:l) do i 
		syms = [p[i] for p in preds]
		ArrayNode(KBEntry(kid, [ex.predicate2id[s] for s in syms]))
	end 
	ds = ProductNode(tuple(xs...))
	(ds, preds)
end

function add_goalstate(ex::LevinASNet, problem, goal = goalstate(ex.domain, problem))
	ex = isspecialized(ex) ? ex : specialize(ex, problem) 
	x = encode_state(ex, goal)
	LevinASNet(ex.domain, ex.type2obs, ex.model_params, ex.predicate2id, ex.action2id, ex.kb, x)
end

function encode_state(ex::LevinASNet, state)
	@assert isspecialized(ex) "Extractor is not specialized for a problem instance"
	x = zeros(Float32, 1, length(ex.predicate2id))
	for p in PDDL.get_facts(state)
		x[ex.predicate2id[p]] = 1
	end
	x
end

#######
#	Define a special LevinState to store actions2id, which is needed for a proper interpretation
#######
struct LevinState{KB, A}
	kb::KB
	action2id::A
end

function Base.show(io::IO, kb::LevinState)
	print(io, "LevinState: (",join(keys(kb.kb), ","),")");
end

function MLUtils.batch(xs::Vector{<:LevinState}) 
	kb = _catobs_kbs([x.kb for x in xs])
	action2id = xs[1].action2id
	@assert all(x.action2id === action2id for x in xs) "creating minibatch from states from multiple different problem instances is not supported."
	LevinState(kb, action2id)
end

function (ex::LevinASNet)(state)
	x = encode_state(ex, state)
	if hasgoal(ex)
		x = vcat(x, ex.goal)
	end 
	kb = ex.kb
	kb = @set kb.kb.pred_1 = x 
	LevinState(kb, ex.action2id)
end


function deduplicate(ls::LevinState)
	LevinState(deduplicate(ls.kb), ls.action2id)
end

#######
#	Define a special LevinModel to output policy and heuristic value
#######
struct LevinModel{PK, KB}
	kbm::KB
end

Flux.@layer LevinModel

function Mill.reflectinmodel(ds::LevinState, fm = d -> Dense(d, 10), fa= SegmentedSumMax; fsm=Dict(), 
	fsa=Dict(), single_key_identity=true, single_scalar_identity=true, all_imputing=false, residual = :none)
	residual == :linear && error("linear residual layer is not supported")

	function isoutput(s::Symbol)
		startswith(string(s),"policy_") && return(true)
		s == :heuristic
	end

	KS = keys(ds.kb)
	kb = atoms(ds.kb)
	layers = NamedTuple{}()
	for k in KS
		predicate = ds.kb[k]
		predicate isa AbstractArray && continue
		m = _reflectinmodel(kb, predicate, fm, fa, isoutput(k) ? fsm : Dict(), k == KS[end] ? fsa : Dict(), "", single_key_identity, single_scalar_identity, all_imputing)[1]
		xx = m(kb, predicate)
		kb = append(kb, k, xx)
		layers = merge(layers, NamedTuple{(k,)}((m,)))
	end

	PK = filter(s -> startswith(string(s),"policy_"), KS)
	kbm = KnowledgeModel(layers)
	LevinModel{PK, typeof(kbm)}(kbm)
end

function (m::LevinModel{KS,KB})(ds::LevinState) where {KS,KB}
	o = _apply_layers(ds.kb, m.kbm)
	heuristic = o[:heuristic]
	policy = vcat(map(k -> reshape(o[k], :, length(heuristic)), KS)...)
	(heuristic, policy)
end


