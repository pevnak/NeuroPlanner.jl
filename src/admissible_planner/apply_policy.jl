struct ApplyPolicy
    model_output_layer
    h_key
end

struct RoundedPolicy
    model_output_layer
    h_key
end

function ApplyPolicy(model; output_key=:o, heuristic_key=:h)
    return ApplyPolicy(model[output_key], heuristic_key)
end

Flux.Functors.@functor ApplyPolicy

function setOutputToPolicy(model, policy::ApplyPolicy; output_key=:o)
    @set model.layer.o = policy
end

function setOutputToPolicy(model; output_key=:o, heuristic_key=:h)
    @set model.layers.o = ApplyPolicy(model; output_key=output_key, heuristic_key=heuristic_key)
end

function removePolicy(model)
    @set model.layers.o = model.layers.o.model_output_layer
end

function (m::ApplyPolicy)(kb::KnowledgeBase, ds)
    p = softmax(m.model_output_layer(kb, ds))

    a= sum(p .* kb[m.h_key], dims=1)
    #a= sum(normed_p .* kb[m.h_key], dims=1)
    
    # return round.(a)
    a
end
Flux.Functors.@functor RoundedPolicy
function RoundedPolicy(model; output_key=:o, heuristic_key=:h)
    if isa(model[output_key], ApplyPolicy) 
        return RoundedPolicy(model[output_key].model_output_layer, heuristic_key)
    end
    return RoundedPolicy(model[output_key], heuristic_key)
end
function RoundedPolicy(m::ApplyPolicy)
    RoundedPolicy(m.model_output_layer, m.h_key)
end
function roundPolicyOutput(model)
    @set model.layers.o = RoundedPolicy(model)
end
function unroundPolictOutput(model)
    @set model.layers.o = ApplyPolicy(model)
end
function (m::RoundedPolicy)(kb::KnowledgeBase, ds)
    p = softmax(m.model_output_layer(kb, ds))
    a= sum(p .* kb[m.h_key], dims=1)
    return round.(a)
end

function boundsoftmaxinputs(out)
    row_diff = diff(out, dims=1)
    #@show row_diff
    upper_bound_check = row_diff .> log(0.95)
    lower_bound_check = row_diff .< log(0.05)

    #@show lower_bound_check
    
    upper_diff = abs.(row_diff[upper_bound_check] .- log(0.95)) .+ 0.00005
    lower_diff = abs.(row_diff[lower_bound_check] .- log(0.05)) .+ 0.00005
    #@show upper_diff
    #@show out[1,vec(upper_bound_check)] + upper_diff /2
    #@show out[2,vec(upper_bound_check)] - upper_diff /2
    #@show diff(out, dims=1) 
    out[1,vec(upper_bound_check)] += upper_diff /2
    out[2,vec(upper_bound_check)] -= upper_diff /2
    #@show diff(out, dims=1)
    #@show diff(out, dims=1) 
    out[1,vec(lower_bound_check)] -= lower_diff /2
    out[2,vec(lower_bound_check)] += lower_diff /2
    #@show out
    out
end
