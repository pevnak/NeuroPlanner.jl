using Test
using NeuroPlanner
using NeuroPlanner: group_facts


# domain_name = ipc23_ferry"
# domain_name = "ipc23_childsnack"
# state = initstate(domain, problem)

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
			if length(ids) > 1 
				ref_ids = only(filter(kv -> kv[1] == f.name,ref.nary))[2]
				@test ids ∈ ref_ids
			elseif length(ids) == 1 
				ref_ids = only(filter(kv -> kv[1] == f.name, ref.unary))[2]
				@test ids ∈ ref_ids
			else
				i = only(findall(f.name .== nullary))
				@test ref.nullary[i] == true
			end
		end
	end
end
