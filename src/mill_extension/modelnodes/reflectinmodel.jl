import Mill: reflectinmodel, _reflectinmodel

function _reflectinmodel(x::AbstractMaskedNode, fm, fa, fsm, fsa, s, ski, args...)
    m, d = _reflectinmodel(x.data, fm, fa, fsm, fsa, s, ski, args...)
    MaskedModel(m), d
end
