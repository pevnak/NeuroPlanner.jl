using Logging
function ffnn(idim, hdim, odim, nlayers)
	nlayers == 1 && return(Dense(idim,odim))
	nlayers == 2 && return(Chain(Dense(idim, hdim, relu), Dense(hdim,odim)))
	nlayers == 3 && return(Chain(Dense(idim, hdim, relu), Dense(hdim, hdim, relu), Dense(odim,odim)))
	error("nlayers should be only in [1,3]")
end

W20AStarPlanner(heuristic::Heuristic; kwargs...) = ForwardPlanner(;heuristic, h_mult=2, kwargs...)
W15AStarPlanner(heuristic::Heuristic; kwargs...) = ForwardPlanner(;heuristic, h_mult=1.5, kwargs...)


function tblogger(filename; min_level::LogLevel=Info, step_increment = 1)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
    logdir = dirname(filename)

    evfile     = open(filename, "w")
    ev_0 = TensorBoardLogger.Event(wall_time=time(), step=0, file_version="brain.Event:2")
    TensorBoardLogger.write_event(evfile, ev_0)

    all_files  = Dict(filename => evfile)
    start_step = 0

    TBLogger{typeof(logdir), typeof(evfile)}(logdir, evfile, all_files, start_step, step_increment, min_level)
end


function dedup_fmb(ds)
    @set ds.x = deduplicate(ds.x)
end

