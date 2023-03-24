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
		ds = multi_predicates(ex, Symbol("x$(i)"), state)
		kb = append(kb, Symbol("x$(i+1)"), ds)
	end
	s = Symbol("x4")
	o = Symbol("o")
	kb = append(kb, o, BagNode(ArrayNode(KBEntry(s, 1:n)), [1:n]))
	addgoal(kb, ex.goal)
end

function add_goalstate(ex::NoProblemNoGoalHE, problem, goal = goalstate(ex.domain, problem))
	ex = specialize(ex, problem)
	add_goalstate(ex, problem, goal)
end

function add_goalstate(ex::ProblemNoGoalHE, problem, goal = goalstate(ex.domain, problem))
	exgoal = ex(goal)
	
	# we need to do this recursively on all layers
	gp = map(enumerate(keys(exgoal.kb))) do (i,k)
		i == 1 && return(exgoal[k])
		i == length(exgoal.kb) && return(exgoal[k])
		eg = exgoal[k]
		ns = map(s -> Symbol("goal_$(s)"), keys(eg.data))
		ProductNode(NamedTuple{ns}(values(eg.data)))
	end
	exgoal = KnowledgeBase(NamedTuple{keys(exgoal)}(tuple(gp...)))

	HyperExtractor(ex.domain, ex.multiarg_predicates, ex.nunanary_predicates, ex.objtype2id, ex.obj2id, exgoal)
end

addgoal(kb::KnowledgeBase, ::Nothing) = kb

function addgoal(kb1::KnowledgeBase{KX,V1}, kb2::KnowledgeBase{KX,V2}) where {KX, V1, V2}
	x = vcat(kb1[:x1], kb2[:x1])
	gp = map(KX[2:end-1]) do k
		ProductNode(merge(kb1[k].data, kb2[k].data))
	end
	KnowledgeBase(NamedTuple{KX}(tuple(x, gp..., kb1.kb[end])))
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

function multi_predicates(ex::HyperExtractor, kid::Symbol, state)
	# Then, we specify the predicates the dirty way
	ks = ex.multiarg_predicates
	xs = map(ex.multiarg_predicates) do k 
		p = ex.domain.predicates[k]
		preds = filter(f -> f.name == k, get_facts(state))
		encode_predicates(p, preds, ex.obj2id, kid)
	end
	ProductNode(NamedTuple{ks}(xs))
end

function encode_predicates(p::PDDL.Signature, preds, obj2id, kid::Symbol)
	xs = map(1:length(p.args)) do i 
		ArrayNode(KBEntry(kid, [obj2id[f.args[i]] for f in preds]))
	end 

	bags = [Int[] for _ in 1:length(obj2id)]
	for (j, f) in enumerate(preds)
		for a in f.args
			push!(bags[obj2id[a]], j)
		end
	end
	BagNode(ProductNode(tuple(xs...)), ScatteredBags(bags))
end