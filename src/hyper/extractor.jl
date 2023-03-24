using PDDL: get_facts, get_args
"""
HyperExtractor{D,G}

Represents PDDL as an HMIL structure.
* Every instance corresponds to one object
* unary and nullary predicates are represented as a property of vertecies
* binary higher-ary predicates will be represented as relations.  
as 
"""
struct HyperExtractor{DO,D,N,G}
	domain::DO
	multiarg_predicates::NTuple{N,Symbol}
	nunanary_predicates::Dict{Symbol,Int64}
	objtype2id::Dict{Symbol,Int64}
	obj2id::D
	goal::G
end

function HyperExtractor(domain)
	dictmap(x) = Dict(reverse.(enumerate(sort(x))))
	predicates = collect(domain.predicates)
	multiarg_predicates = tuple([kv[1] for kv in predicates if length(kv[2].args) > 1]...)
	nunanary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) ≤  1])
	objtype2id = Dict(s => i + length(nunanary_predicates) for (i, s) in enumerate(collect(keys(domain.typetree))))
	HyperExtractor(domain, multiarg_predicates, nunanary_predicates, objtype2id, nothing, nothing)
end

NoProblemNoGoalHE{DO,N} = HyperExtractor{DO,Nothing,N,Nothing} where {DO,N}
ProblemNoGoalHE{DO,N} = HyperExtractor{DO,D,N,Nothing} where {DO,D<:Dict,N}

function HyperExtractor(domain, problem; embed_goal = true)
	ex = HyperExtractor(domain)
	specialize(ex, problem; embed_goal)
end

"""
specialize(ex::HyperExtractor{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(ex::HyperExtractor, problem)
	obj2id = Dict(v => i for (i, v) in enumerate(problem.objects))
	HyperExtractor(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id,  obj2id, nothing)
end

function (ex::HyperExtractor)(state::GenericState)
	length(state.types) != length(ex.obj2id) && error("number of objects in state and problem instance does not match")
	x = nunary_predicates(ex, state)
	kb = KnowledgeBase((;x1 = x))
	n = size(x,2)
	for i in 1:3
		s = Symbol("x$(i)")
		o = Symbol("x$(i+1)")
		ds = multi_predicates(ex, ArrayNode(KBEntry(s, 1:n)), state)
		kb = append(kb, o, ds)
	end
	s = Symbol("x4")
	o = Symbol("x5")
	append(kb, o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))
end

function add_goalstate(ex::NoProblemNoGoalHE, problem, goal = goalstate(ex.domain, problem))
	ex = specialize(ex, problem)
	add_goalstate(ex, problem, goal)
end

function add_goalstate(ex::ProblemNoGoalHE, problem, goal = goalstate(ex.domain, problem))
	exgoal = ex(goal)
	
	# we need to do this recursively on all layers
	# ns = map(s -> Symbol("goal_$(s)"), keys(exgoal.data.data))
	# exgoal = BagNode(ProductNode(NamedTuple{ns}(values(exgoal.data.data))), exgoal.bags)

	HyperExtractor(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.obj2id, exgoal)
end

"""
nunary_predicates(ex::HyperExtractor, state)

Create matrix with one column per object and encode by one-hot-encoding unary predicates 
and types of objects. Nunary predicates are encoded as properties of all objects.
"""
function nunary_predicates(ex::HyperExtractor, state)
	# first, we completely specify the matrix with properties
	x = zeros(Float32, length(ex.nunanary_predicates) + length(ex.objtype2id), length(ex.obj2id))
	for s in state.types
		i = ex.objtype2id[s.name]
		j = ex.obj2id[only(s.args)]
		x[i, j] = 1
	end

	for f in filter(f -> length(get_args(f)) < 2, get_facts(state))
		a = get_args(f)
		pid = ex.nunanary_predicates[f.name]
		if length(a) == 1 
			vid = ex.obj2id[only(get_args(f))]
			x[pid, vid] = 1
		else length(a) == 0
			x[pid,:] .= 1
		end
	end
	x
end

"""
multi_predicates(ex::HyperExtractor, x, state)

Create and HMIL structure encoding predicates with arities higher than 2 as ProductNodes.
Each object is represented as a collection of predicates it participates in.
"""
function multi_predicates(ex::HyperExtractor, x, state)
	# Then, we specify the predicates the dirty way
	all_predicates = [Dict{Symbol, Any}() for _ in 1:length(state.types)]
	for f in filter(f -> length(get_args(f)) ≥ 2, get_facts(state))
		ii = [ex.obj2id[i] for i in get_args(f)]	#arguments to ids
		p = _predicate(x, ii)
		# We add the predicate to all of its objects
		for i in ii 
			d = all_predicates[i]
			if haskey(d, f.name)
				push!(d[f.name], p)
			else
				d[f.name] = [p]
			end
		end
	end
	# each new representation of a vertex is composed from a bags of predicates of 
	# each type 
	pnames = ex.multiarg_predicates
	xs = map(all_predicates) do predicates
		xx = map(pnames) do pname
			if !haskey(predicates, pname) 
				xx = _empty_predicate(ex.domain.predicates[pname], x)
				return(BagNode(xx, [0:-1])) # Possibly replace missing with an empty arrity
			else
				xs = predicates[pname]
				return(BagNode(reduce(catobs, xs), [1:length(xs)]))
			end
		end
		ProductNode(NamedTuple{pnames}(xx))
	end
	reduce(catobs, xs)
end

function _predicate(xx::AbstractMillNode, ii)
	xs = [xx[i] for i in ii]
	ProductNode(tuple(xs...))
end

function _empty_predicate(p::PDDL.Signature, xx::AbstractMillNode)
	xs = [xx[0:-1] for _ in 1:length(p.argtypes)]
	ProductNode(tuple(xs...))
end