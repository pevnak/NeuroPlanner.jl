struct ASNet{DO,TO,P<:Union{Dict,Nothing},KB<:Union{Nothing, KnowledgeBase},G<:Union{Nothing,Matrix}}
	domain::DO
	type2obs::TO
	model_params::NamedTuple{(:message_passes, :residual), Tuple{Int64, Symbol}}
	predicate2id::P
	kb::KB
	goal::G
end

isspecialized(ex::ASNet) = (ex.predicate2id !== nothing) && (ex.kb !== nothing)
hasgoal(ex::ASNet) = ex.goal !== nothing

function ASNet(domain; message_passes = 2, residual = :linear)
	model_params = (;message_passes, residual)
	ASNet(domain, nothing, model_params, nothing, nothing, nothing)
end

function Base.show(io::IO, ex::ASNet)
	s = isspecialized(ex) ? "Specialized" : "Unspecialized"
	s *=" ASNet for $(ex.domain.name )"
	if isspecialized(ex)
		s *= " ($(length(ex.predicate2id)))"
	end
	print(io, s)
end

function specialize(ex::ASNet, problem)
	# map containing lists of objects of the same type
	type2obs = type2objects(ex.domain, problem)

	# create a map mapping predicates to their ID, which is pr
	predicates = mapreduce(union, values(ex.domain.predicates)) do p 
		allgrounding(problem, p, type2obs)
	end
	predicate2id = Dict(reverse.(enumerate(predicates)))
	ex = ASNet(ex.domain, type2obs, ex.model_params, predicate2id, nothing, nothing)

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

	ASNet(ex.domain, type2obs, ex.model_params, predicate2id, kb, nothing)
end

"""
type2objects(domain, problem)

create a map of types to `problem.objects` including the type hierarchy 

example
```julia
julia> domain.typetree
Dict{Symbol, Vector{Symbol}} with 6 entries:
  :count                  => []
  Symbol("fast-elevator") => []
  :elevator               => [Symbol("slow-elevator"), Symbol("fast-elevator")]
  :passenger              => []
  :object                 => [:elevator, :passenger, :count]
  Symbol("slow-elevator") => []


julia> problem.objtypes
Dict{Const, Symbol} with 19 entries:
  fast0   => Symbol("fast-elevator")
  p0      => :passenger
  n4      => :count
  slow1-0 => Symbol("slow-elevator")
  p1      => :passenger
  p2      => :passenger
  p4      => :passenger
  n1      => :count
  p3      => :passenger
  n3      => :count
  slow0-0 => Symbol("slow-elevator")
  n0      => :count
  n2      => :count

julia> type2objects(domain, problem)
Dict{Symbol, Vector{Const}} with 6 entries:
  :count                  => [n4, n1, n3, n0, n2]
  Symbol("fast-elevator") => [fast0, fast1]
  :passenger              => [p0, p1, p2, p4, p3]
  :elevator               => [slow1-0, slow0-0, fast0, fast1]
  Symbol("slow-elevator") => [slow1-0, slow0-0]
  :object                 => [slow1-0, slow0-0, fast0, fast1, p0, p1, p2, p4, p3, n…
```
"""
function type2objects(domain, problem)
	#first create a direct descendants
	type2obs = map(unique(values(problem.objtypes))) do k 
		k => [n for (n,v) in problem.objtypes if v == k]
	end |> Dict

	kv = sort(collect(domain.typetree), lt = (i,j) -> i[1] ∈ j[2])
	for (k, v) in kv
		isempty(v) && continue
		type2obs[k] = reduce(union, [type2obs[i] for i in v])
	end
	type2obs
end

function add_goalstate(ex::ASNet, problem, goal = goalstate(ex.domain, problem))
	ex = isspecialized(ex) ? ex : specialize(ex, problem) 
	x = encode_input(ex, goal)
	ASNet(ex.domain, ex.type2obs, ex.model_params, ex.predicate2id, ex.kb, x)
end

function (ex::ASNet)(state)
	x = encode_input(ex::ASNet, state)
	if hasgoal(ex)
		x = vcat(x, ex.goal)
	end 
	kb = ex.kb
	kb = @set kb.kb.x1 = x 
	kb
end

function encode_input(ex::ASNet, state)
	@assert isspecialized(ex) "Extractor is not specialized for a problem instance"
	x = zeros(Float32, 1, length(ex.predicate2id))
	for p in PDDL.get_facts(state)
		x[ex.predicate2id[p]] = 1
	end
	x
end

function encode_actions(ex::ASNet, kid::Symbol)
	actions = ex.domain.actions
	ns = tuple(keys(actions)...)
	xs = map(values(ex.domain.actions)) do action
		encode_action(ex, action, kid)
	end
	length(xs) == 1 ? only(xs) : ProductNode(NamedTuple{ns}(tuple(xs...)))
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
allgrounding(action, type2obs)

create all possible grounding of predicates in `action` while assuming objects with types in `type2obs` 
"""
function allgrounding(action::GenericAction, type2obs::Dict)
	predicates = extract_predicates(action)
	allgrounding(action, predicates, type2obs)
end

function allgrounding(action::GenericAction, predicates::Vector{<:Term}, type2obs::Dict)
	types = [type2obs[k] for k in action.types]
	assignments = vec(collect(Iterators.product(types...)))
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

function ground(p::Const, assignment::Dict)
	p
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

function extract_predicates(p::Const)
	[p] 
end

