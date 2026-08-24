# Julia-side performance for the DifferLand DALEC990 port.
#
# Reports, per config: forward-only, loss (forward + reductions), value+gradient
# (Enzyme reverse), and one full Adam calibration iteration. Compile/TTFX time is
# measured and reported separately, never folded into the steady-state numbers.
#
# Run single-threaded and pinned:  taskset -c N julia -t 1 --project=. bench/bench_julia.jl
using DifferLand, StaticArrays, Enzyme, Printf, JSON, Statistics
const FIX = joinpath(@__DIR__, "..", "fixtures")
const CALIB_ITERS = parse(Int, get(ENV, "CALIB_ITERS", "2000"))
const OUTJSON = get(ENV, "OUTJSON", joinpath(@__DIR__, "..", "results", "julia_cpu.json"))

"""min-of-reps timing with a fixed wall-clock budget, after warmup."""
function timeit(f!; budget = 3.0, minreps = 5)
    f!()                                   # warm
    best = Inf; n = 0; t0 = time()
    while n < minreps || time() - t0 < budget
        t = time_ns(); f!(); dt = (time_ns() - t) / 1e9
        best = min(best, dt); n += 1
        n > 100_000 && break
    end
    return best, n
end

function main()
    site, cfgs, k = load_fixture(FIX)
    res = Dict{String,Any}("host" => gethostname(), "nthreads" => Threads.nthreads(),
                           "julia" => string(VERSION), "T" => site.T,
                           "cpu" => Sys.CPU_NAME, "configs" => Dict{String,Any}())
    @printf("julia %s  threads=%d  cpu=%s  T=%d\n\n",
            VERSION, Threads.nthreads(), Sys.CPU_NAME, site.T)

    for c in cfgs
        st = c.st; np = length(c.θ)
        out  = zeros(Float32, 32, site.T); dout = zeros(Float32, 32, site.T)
        θ    = copy(c.θ);                  g    = zeros(Float32, np)

        # ---- compile (TTFX) measured on a fresh call of each entry point ----
        tc_fwd  = @elapsed forward!(out, θ, site, st)
        tc_loss = @elapsed compute_loss!(out, θ, site, st, k)
        tc_grad = @elapsed value_and_grad!(g, out, dout, θ, site, st, k)

        t_fwd,  n1 = timeit(() -> forward!(out, θ, site, st))
        t_loss, n2 = timeit(() -> compute_loss!(out, θ, site, st, k))
        t_grad, n3 = timeit(() -> value_and_grad!(g, out, dout, θ, site, st, k))

        # ---- end-to-end: CALIB_ITERS real Adam iterations from the fixture point ----
        θc = copy(c.θ); s = Adam(np; lr = 5f-4)
        losses = Float32[]
        calib_step!(θc, g, out, dout, site, st, k, s)              # warm + 1st iter
        θc = copy(c.θ); s = Adam(np; lr = 5f-4)
        tcal = time_ns()
        for i in 1:CALIB_ITERS
            L = calib_step!(θc, g, out, dout, site, st, k, s)
            (i <= 20 || i % 500 == 0) && push!(losses, L)
        end
        tcal = (time_ns() - tcal) / 1e9

        @printf("--- %s (%d params) ---\n", c.name, np)
        @printf("  forward only            %8.3f ms   (min of %d)\n", t_fwd*1e3,  n1)
        @printf("  loss (fwd + reductions) %8.3f ms   (min of %d)\n", t_loss*1e3, n2)
        @printf("  value+grad (Enzyme rev) %8.3f ms   (min of %d)   overhead vs loss %.2fx\n",
                t_grad*1e3, n3, t_grad/t_loss)
        @printf("  Adam iteration (e2e)    %8.3f ms   (%d iters in %.2f s)\n",
                tcal/CALIB_ITERS*1e3, CALIB_ITERS, tcal)
        @printf("  -> 25,000-iter calibration: %.1f s\n", tcal/CALIB_ITERS*25_000)
        @printf("  compile (TTFX): fwd %.2f s  loss %.2f s  grad %.2f s\n",
                tc_fwd, tc_loss, tc_grad)
        @printf("  loss after %d iters: %.6f (started %.6f)\n\n",
                CALIB_ITERS, losses[end], losses[1])

        res["configs"][c.name] = Dict(
            "nparams" => np, "forward_ms" => t_fwd*1e3, "loss_ms" => t_loss*1e3,
            "grad_ms" => t_grad*1e3, "grad_over_loss" => t_grad/t_loss,
            "adam_iter_ms" => tcal/CALIB_ITERS*1e3,
            "calib_25k_s" => tcal/CALIB_ITERS*25_000,
            "compile_s" => Dict("forward" => tc_fwd, "loss" => tc_loss, "grad" => tc_grad),
            "calib_iters" => CALIB_ITERS, "loss_trace" => losses)
    end
    mkpath(dirname(OUTJSON))
    open(OUTJSON, "w") do io; JSON.print(io, res, 2); end
    println("wrote ", OUTJSON)
end
main()
