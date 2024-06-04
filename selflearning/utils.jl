function ffnn(idim, hdim, odim, nlayers)
	nlayers == 1 && return(Dense(idim,odim))
	nlayers == 2 && return(Chain(Dense(idim, hdim, relu), Dense(hdim,odim)))
	nlayers == 3 && return(Chain(Dense(idim, hdim, relu), Dense(hdim, hdim, relu), Dense(hdim,odim)))
	error("nlayers should be only in [1,3]")
end

W20AStarPlanner(heuristic::Heuristic; kwargs...) = ForwardPlanner(;heuristic, h_mult=2, kwargs...)
W15AStarPlanner(heuristic::Heuristic; kwargs...) = ForwardPlanner(;heuristic, h_mult=1.5, kwargs...)

function dedup_fmb(ds)
    @set ds.x = deduplicate(ds.x)
end

