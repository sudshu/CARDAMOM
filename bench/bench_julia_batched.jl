# Julia throughput across independent restarts -- the matched comparison to the
# JAX vmap sweep. Upstream runs 40 random restarts per (site, config); those are
# embarrassingly parallel, so the honest question is not "one core vs one A100"
# but "members per second per machine".
#
# Each thread owns its own theta / gradient / (32,T) buffer / shadow; `site` is
# shared read-only. Run as: taskset -c <list> julia -t N --project=. bench/bench_julia_batched.jl
using DifferLand, Enzyme, Printf, JSON, Base.Threads
const FIX = joinpath(@__DIR__, "..", "fixtures")
const B = parse(Int, get(ENV, "BATCH", "512"))
const OUTJSON = joinpath(@__DIR__, "..", "results", "julia_batched_t$(nthreads()).json")

function main()
    site, cfgs, k = load_fixture(FIX)
    res = Dict{String,Any}("nthreads" => nthreads(), "batch" => B, "T" => site.T,
                           "configs" => Dict{String,Any}())
    @printf("julia %s  threads=%d  batch=%d  T=%d\n", VERSION, nthreads(), B, site.T)

    for c in cfgs
        np = length(c.θ)
        # per-member state: slightly perturbed parameters, as in the vmap sweep
        Θ = [Float32.(c.θ .* (1 .+ 1f-4 .* randn(Float32, np))) for _ in 1:B]
        G = [zeros(Float32, np) for _ in 1:B]
        # explicit chunking rather than threadid() indexing: in julia 1.12
        # threadid() can exceed nthreads() (interactive pool), and :dynamic tasks
        # may migrate. One buffer pair per chunk is correct either way.
        chunks = collect(Iterators.partition(1:B, cld(B, nthreads())))
        OUT  = [zeros(Float32, 32, site.T) for _ in eachindex(chunks)]
        DOUT = [zeros(Float32, 32, site.T) for _ in eachindex(chunks)]

        run! = function ()
            @sync for (ci, rng) in enumerate(chunks)
                Threads.@spawn begin
                    o = OUT[ci]; d = DOUT[ci]
                    for i in rng
                        value_and_grad!(G[i], o, d, Θ[i], site, c.st, k)
                    end
                end
            end
        end
        run!()                                          # warm / compile
        best = Inf
        for _ in 1:3
            t = time_ns(); run!(); best = min(best, (time_ns() - t) / 1e9)
        end
        @printf("  %-9s B=%4d  %9.2f ms total  %8.4f ms/member  (%d threads, %d chunks)\n",
                c.name, B, best*1e3, best*1e3/B, nthreads(), length(chunks))
        res["configs"][c.name] = Dict("total_ms" => best*1e3,
                                      "per_member_ms" => best*1e3/B, "nparams" => np,
                                      "nchunks" => length(chunks))
    end
    mkpath(dirname(OUTJSON))
    open(OUTJSON, "w") do io; JSON.print(io, res, 2); end
    println("wrote ", OUTJSON)
end
main()
