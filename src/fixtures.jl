"""Loader for the fixture exported by bench/export_fixture.py.

Binaries are flat little-endian float32 in numpy C order, so a numpy (T, 18)
array read into a Julia (18, T) array is exactly its transpose -- which is the
layout `Site` wants. Same trick maps a numpy (n_in, n_out) weight matrix onto
the Julia (n_out, n_in) matrix used for `Wt * x`.
"""
struct FixtureConfig
    name::String
    st::StressType
    θ::Vector{Float32}
    ref_loss::Float32
    ref_outT::Matrix{Float32}     # (32, T)
    ref_grad::Vector{Float32}     # same layout as θ
end

_rd(dir, name, dims...) = read!(joinpath(dir, name * ".bin"),
                                Array{Float32}(undef, dims...))

stress_type(name::AbstractString) =
    name == "default" ? Default() :
    name == "nn_whole" ? NNWhole() :
    error("unsupported config $name")

function load_fixture(dir::AbstractString)
    man = JSON.parsefile(joinpath(dir, "manifest.json"))
    T = Int(man["T"]); train_end = Int(man["train_end_idx"])
    k = Float32(man["k"]); ce_opt = Float32(man["ce_opt"])

    bnd = Bounds(_rd(dir, "parmin_arr", 38), _rd(dir, "parmax_arr", 38))
    site = Site(_rd(dir, "met", 18, T), _rd(dir, "target", 10, T),
                train_end, ce_opt, bnd)

    cfgs = FixtureConfig[]
    for (name, c) in man["configs"]
        st = stress_type(name)
        nl = length(c["mlp"])
        θ = Float32[]
        append!(θ, _rd(dir, "$(name)_param_initial", 30))
        append!(θ, _rd(dir, "$(name)_pool_initial", 8))
        g = Float32[]
        append!(g, _rd(dir, "$(name)_grad_param", 30))
        append!(g, _rd(dir, "$(name)_grad_pool", 8))
        for li in 0:(nl-1)
            ws = Int.(c["mlp"][li+1]["weights"]); bs = Int.(c["mlp"][li+1]["biases"])
            # numpy (n_in, n_out) -> flat C order; that IS Julia (n_out, n_in) col-major
            append!(θ, vec(_rd(dir, "$(name)_W$(li)", ws[2], ws[1])))
            append!(θ, _rd(dir, "$(name)_b$(li)", bs[1]))
            append!(g, vec(_rd(dir, "$(name)_grad_W$(li)", ws[2], ws[1])))
            append!(g, _rd(dir, "$(name)_grad_b$(li)", bs[1]))
        end
        @assert length(θ) == nparams(st) "θ length $(length(θ)) != $(nparams(st)) for $name"
        push!(cfgs, FixtureConfig(name, st, θ, Float32(c["ref_loss"]),
                                  _rd(dir, "$(name)_output", 32, T), g))
    end
    return site, cfgs, k
end
