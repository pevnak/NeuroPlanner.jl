struct KnowledgeModel{KS, VS}
	layers::NamedTuple{KS,VS}
end

Flux.functor(m::KnowledgeModel{KS,VS}) where {KS,VS} = ((m.layers...), m -> MultiheadAttention{KS,VS}(m...))

function Base.show(io::IO, km::KnowledgeModel) 
	print(io, "KnowledgeModel: (",join(keys(km), ","),")");
end

Base.getindex(km::KnowledgeModel, k::Symbol) = km.layers[k]
Base.keys(km::KnowledgeModel) = keys(km.layers)

function reflectinmodel(ds::KnowledgeBase, fm=d -> Dense(d, 10), fa= BagCount ∘ SegmentedMeanMax; fsm=Dict(),
        fsa=Dict(), single_key_identity=true, single_scalar_identity=true, all_imputing=false)
	kb = atoms(ds)
	layers = NamedTuple{}()
	for k in keys(ds)[2:end]
		predicate = ds[k]
		predicate isa AbstractArray && continue
		m = _reflectinmodel(kb, predicate, fm, fa, fsm, fsa, "", single_key_identity, single_scalar_identity, all_imputing)[1]
		xx = m(kb, predicate)
		kb = append(kb, k, xx)
		layers = merge(layers, NamedTuple{(k,)}((m,)))
	end
	KnowledgeModel(layers)
end

function (m::KnowledgeModel{KS,VS})(kb::KnowledgeBase) where {KS,VS}
	o = atoms(kb)
	o = _apply_layers(o, kb, m)
	o[last(KS)]
end

function _apply_layers(o::KnowledgeBase, kb::KnowledgeBase, m::KnowledgeModel{KS, VS}) where {KS,VS}
	for k in KS
		o = append(o, k, m[k](o, kb.kb[k]))
	end
	o
end

# @generated function _apply_layers(o::KnowledgeBase, kb::KnowledgeBase, m::KnowledgeModel{KS, VS}) where {KS,VS}
#     statements = map(KS) do k 
#     	quote
# 			xx = m.layers.$k(o, kb.kb.$k)
# 			o = append(o, :($k), xx)
# 		end
# 	end
#     quote
#         $(statements...)
#     end
# end
