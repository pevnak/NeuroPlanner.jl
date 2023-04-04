struct ASNet{DO,TO, P}
	domain::D
	type2obs::TO
	model_params::NamedTuple{(:message_passes, :residual), Tuple{Int64, Symbol}}
	predicate2id::P
	goal::G
end


function ASNet(domain; message_passes = 2, residual = :linear)
	model_params = (;message_passes, residual)
	type2obs = map(unique(values(problem.objtypes))) do k 
		k => [n for (n,v) in problem.objtypes if v == k]
	end |> Dict

end


function specialize(domain, problem; add_goal = true)
	# create a map mapping predicates to their ID, which is pr
	predicates = mapreduce(union, values(domain.predicates)) do p 
		allgrounding(problem, p, ex.type2obs)
	end
	predicate2id = Dict(reverse.(enumerate(predicates)))

	# just add fake input for a se to have something, it does not really matter what
	n = length(predicate2id)
	x = zeros(Float32, 0, n)
	kb = KnowledgeBase((;x1 = x))
	sₓ = :x1
	for i in 1:ex.model_params.message_passes
		input_to_gnn = last(keys(kb))
		ds = encode_actions(ex, input_to_gnn)
		kb = append(kb, layer_name(kb, "gnn"), ds)
		if residual !== :none #if there is a residual connection, add it 
			kb = add_residual_layer(kb, keys(kb)[end-1:end], n)
		end
	end
	s = last(keys(kb))
	kb = append(kb, :o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))
	addgoal(ex, kb)
end

function encode_actions(ex::ASNet, kid::Symbol)
	xs = mapreduce(merge, collect(domain.actions)) do kv 
		name, action = kv 
		NamedTuple{(name),}((encode_action(ex, action, kid)),)
	end
	ProductNode(xs)
end

function encode_action(ex::ASNet, action::GenericAction, kid::Symbol)
	preds = allgrounding(action, ex.type2obs)
	encode_predicates(ex, preds, kid)
end


function encode_predicates(ex::ASNet, preds, kid::Symbol)
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

"""
allgrounding(problem::GenericProblem, p::Signature, type2obs::Dict)

create all possible groundings of a predicate `p` for a `problem` 
"""
function allgrounding(problem::GenericProblem, p::PDDL.Signature, type2obs::Dict)
	combinations = Iterators.product([type2obs[k] for k in p.argtypes]...)
	map(combinations) do args
		Julog.Compound(p.name, [a for a in args])
	end |> vec
end

function allgrounding(problem::GenericProblem, p::PDDL.Signature)
	type2obs = map(unique(values(problem.objtypes))) do k 
		k => [n for (n,v) in problem.objtypes if v == k]
	end |> Dict
	allgrounding(problem, p, type2obs)
end


"""
allgrounding(action, type2obs; unique_args = true)

create all possible grounding of predicates in `action` while assuming objects with types in `type2obs` 
"""
function allgrounding(action::GenericAction, type2obs::Dict; unique_args = true)
	# for all possible grounding of arguments, ground the predicates
	predicates = extract_predicates(action)
	types = [type2obs[k] for k in action.types]
	assignments = vec(collect(Iterators.product(types...)))
	unique_args && filter!(v -> length(unique(v)) == length(v), assignments) 
	as = map(assignments) do v 
		assignment = Dict(zip(action.args, v))
		[ground(p, assignment) for p in predicates]
	end 
end

"""
ground(p, assignement)

ground variables in predicate `p` with assignment 
"""
function ground(p::Compound, assignment::Dict)
	Compound(p.name, [assignment[k] for k in p.args])
end

"""
extract_predicates(action)

extract all predicates from action while ommiting logical operators (:and, :not, :or)
"""
function extract_predicates(action::GenericAction)
	union(extract_predicates(action.precond), extract_predicates(action.effect))
end

function extract_predicates(p::Compound)
	p.name ∈ (:and, :not, :or) && return(reduce(union, extract_predicates.(p.args)))
	[p] 
end

