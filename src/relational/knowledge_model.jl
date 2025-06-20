struct KnowledgeModel{KS, VS}
	layers::NamedTuple{KS,VS}
end

Flux.@layer KnowledgeModel

function Base.show(io::IO, km::KnowledgeModel) 
	print(io, "KnowledgeModel: (",join(keys(km), ","),")");
end

Base.getindex(km::KnowledgeModel, k::Symbol) = km.layers[k]
Base.keys(km::KnowledgeModel) = keys(km.layers)

function reflectinmodel(ds::KnowledgeBase{KS,VS}, fm = d -> Dense(d, 10), fa= SegmentedSumMax; fsm=Dict(), 
	fsa=Dict(), single_key_identity=true, single_scalar_identity=true, all_imputing=false, residual = :dense) where {KS,VS}
	kb = atoms(ds)
	layers = NamedTuple{}()
	for k in KS
		predicate = ds[k]
		predicate isa AbstractArray && continue
		m = _reflectinmodel(kb, predicate, fm, fa, k == KS[end] ? fsm : Dict(), k == KS[end] ? fsa : Dict(), "", single_key_identity, single_scalar_identity, all_imputing)[1]
		xx = m(kb, predicate)
		if startswith(String(k),"res_") && residual == :linear # kind of suboptimal, since we recompute, but good enough
			odim = size(xx,1)
			m = _reflectinmodel(kb, predicate, d -> identity, fa, k == KS[end] ? fsm : Dict("" => d -> Dense(d,odim)), k == KS[end] ? fsa : Dict(), "", single_key_identity, single_scalar_identity, all_imputing)[1]		
			xx = m(kb, predicate)
		end
		kb = append(kb, k, xx)
		layers = merge(layers, NamedTuple{(k,)}((m,)))
	end
	KnowledgeModel(layers)
end

function (m::KnowledgeModel{KS,VS})(kb::KnowledgeBase) where {KS,VS}
	o = _apply_layers(kb, m)
	o[last(KS)]
end

_apply_layers(kb::KnowledgeBase, m::KnowledgeModel) = _apply_layers(atoms(kb), kb, m.layers)

function _apply_layers(o::KnowledgeBase, kb::KnowledgeBase, layers::NamedTuple{KS, VS}) where {KS,VS}
	isempty(layers) && return(o)
	k = first(KS)
	o = append(o, k, layers[k](o, kb.kb[k]))
	_apply_layers(o, kb, Base.tail(layers))
end
