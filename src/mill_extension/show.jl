import Mill: _levelparams, _show_submodels

_levelparams(m::MaskedModel) = Flux.params(m.m)

# _show_submodels(io, m::MaskedModel) = print(io, " ↦ ", m.m)
