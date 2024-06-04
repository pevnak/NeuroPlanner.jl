using Test
using NeuroPlanner
using NeuroPlanner: group_facts


# domain_name = "ipc23_ferry"
# domain_name = "ipc23_childsnack"
# state = initstate(domain, problem)

@testset "Conversion of states to intstates" begin
	@testset "Domain: $domain_name" for domain_name in DOMAINS
		domain, problem = load_problem_domain(domain_name) 
		state = initstate(domain, problem)
		objs = sort([k.name for k in collect(keys(problem.objtypes))])
		for k in keys(domain.constypes)
			push!(objs, k.name)
		end
		obj2id = Dict([k => i for (i, k) in enumerate(objs)])
		pifo = NeuroPlanner.PredicateInfo(domain)
		facts = collect(PDDL.get_facts(state))

		# we verify that all facts are in groupped facts, which means 
		# that we effectively write anothe slow parser
		ref = NeuroPlanner.intstates(domain, obj2id, facts)
	    nullary = collect(filter(k -> isempty(domain.predicates[k].args), keys(domain.predicates)))
		for f in facts
			ids = tuple([obj2id[a.name] for a in f.args]...)
			pid = findfirst(f.name .==  pifo.predicates)
			s = NeuroPlanner.IntState(pid, ids)
			@test s ∈ ref[length(ids) + 1]
		end
	end
end

@testset "group_facts" begin
	@testset "Domain: $domain_name" for domain_name in DOMAINS
		domain, problem = load_problem_domain(domain_name) 
		state = initstate(domain, problem)
		objs = sort([k.name for k in collect(keys(problem.objtypes))])
		for k in keys(domain.constypes)
			push!(objs, k.name)
		end
		obj2id = Dict([k => i for (i, k) in enumerate(objs)])
		facts = collect(PDDL.get_facts(state))

		# we verify that all facts are in groupped facts, which means 
		# that we effectively write anothe slow parser
		ref = NeuroPlanner.group_facts(domain, obj2id, facts)
	    nullary = collect(filter(k -> isempty(domain.predicates[k].args), keys(domain.predicates)))
		for f in facts
			ids = tuple([obj2id[a.name] for a in f.args]...)
			length(ids) > 3 && error("$(length(ids))-ary predicate exists")
			if length(ids) == 3
				ref_ids = only(filter(kv -> kv[1] == f.name,ref.ternary))[2]
				@test ids ∈ ref_ids
			elseif length(ids) == 2
				ref_ids = only(filter(kv -> kv[1] == f.name,ref.binary))[2]
				@test ids ∈ ref_ids
			elseif length(ids) == 1 
				ref_ids = only(filter(kv -> kv[1] == f.name, ref.unary))[2]
				@test only(ids) ∈ ref_ids
			else
				i = only(findall(f.name .== nullary))
				@test ref.nullary[i] == true
			end
		end
	end
end

@testset "Group facts for AtomBinary extractor" begin
	function group_facts_ref(ex::AtomBinary, facts)
	    NT = @NamedTuple{position::Int64, atom_id::Int64}
	    ids_in_facts = [NT[] for _ in 1:length(ex.obj2id)]
	    for (i, a) in enumerate(facts)
	        for (j,o) in enumerate(a.args)
	            oid = ex.obj2id[o.name]
	            push!(ids_in_facts[oid], (;position = j, atom_id = i))
	        end
	    end
	    ids_in_facts
	end

	@testset "Domain: $domain_name" for domain_name in DOMAINS
		domain, problem = load_problem_domain(domain_name) 
		pddld = AtomBinaryME(domain; message_passes = graph_layers, residual)
		ex, state = initproblem(pddld, problem);
		facts = collect(PDDL.get_facts(state))
		@test sort.(NeuroPlanner.group_facts(ex, facts)) == sort.(group_facts_ref(ex, facts))
	end 
end
