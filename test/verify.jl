# Verification gate for the Julia port of DifferLand's DALEC990.
#
# Bare agreement in float32 is not a meaningful bar on its own: the model is
# float32-chaotic over a 3287-step sequential chain (jax-f32 differs from
# jax-f64 by ~5e-3 on the trajectory and ~9e-4 on the gradient). So the gate is
# relative: Julia must (a) agree with jax-f32 far better than jax-f32 agrees
# with jax-f64, and (b) be no further from the f64 truth than jax-f32 is.
using DifferLand, StaticArrays, Enzyme, Printf, JSON
const FIX = joinpath(@__DIR__, "..", "fixtures")

relerr(a, b) = abs(a - b) / max(abs(b), 1e-30)

"""|a-b| / max(|b|, RMS of that row of b) -- the DALEC_1100 C-vs-JAX bar."""
function scaled_maxerr(A::AbstractMatrix, B::AbstractMatrix)
    worst = 0.0; at = (0, 0)
    for j in 1:size(A, 1)
        rms = sqrt(sum(abs2, @view B[j, :]) / size(B, 2))
        for t in 1:size(A, 2)
            e = abs(A[j, t] - B[j, t]) / max(abs(B[j, t]), rms, 1e-30)
            if e > worst; worst = e; at = (j, t); end
        end
    end
    return worst, at
end

l2rel(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))
cosim(a, b) = (a' * b) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)))

function main()
    site, cfgs, k = load_fixture(FIX)
    f64ref = JSON.parsefile(joinpath(FIX, "f64_reference.json"))
    @printf("site US-Var: T=%d train_end=%d ce_opt=%.4f k=%.1f\n\n",
            site.T, site.train_end, site.ce_opt, k)
    allpass = true

    for c in cfgs
        @printf("========== config %-9s (%d trainable params) ==========\n",
                c.name, length(c.θ))
        out = zeros(Float32, 32, site.T)
        L = compute_loss!(out, c.θ, site, c.st, k)
        dθ = zeros(Float32, length(c.θ)); dout = zeros(Float32, 32, site.T)
        Enzyme.autodiff(set_runtime_activity(Reverse), compute_loss!,
                        Duplicated(out, dout), Duplicated(c.θ, dθ),
                        Const(site), Const(c.st), Const(k))

        o64 = Array{Float64}(undef, 32, site.T)
        read!(joinpath(FIX, "$(c.name)_output_f64.bin"), o64)
        g64 = Array{Float64}(undef, length(c.θ))
        read!(joinpath(FIX, "$(c.name)_grad_f64.bin"), g64)
        L64 = Float64(f64ref[c.name]["loss_f64"])

        jl_vs_jax_traj, at = scaled_maxerr(out, c.ref_outT)
        jl_vs_f64_traj, _  = scaled_maxerr(Float64.(out), o64)
        jax_vs_f64_traj    = Float64(f64ref[c.name]["traj_scaled_maxerr"])
        jl_vs_jax_g  = l2rel(Float64.(dθ), Float64.(c.ref_grad))
        jl_vs_f64_g  = l2rel(Float64.(dθ), g64)
        jax_vs_f64_g = Float64(f64ref[c.name]["grad_relerr"])

        println("                            jl-f32 v jax-f32   jl-f32 v f64   jax-f32 v f64")
        @printf("loss (rel)                      %.3e        %.3e      %.3e\n",
                relerr(Float64(L), Float64(c.ref_loss)), relerr(Float64(L), L64),
                Float64(f64ref[c.name]["loss_relerr"]))
        @printf("trajectory (scaled max)         %.3e        %.3e      %.3e\n",
                jl_vs_jax_traj, jl_vs_f64_traj, jax_vs_f64_traj)
        @printf("gradient (L2 rel)               %.3e        %.3e      %.3e\n",
                jl_vs_jax_g, jl_vs_f64_g, jax_vs_f64_g)
        @printf("gradient cosine vs jax-f32: %.10f   worst traj cell: col %d, t %d\n",
                cosim(Float64.(dθ), Float64.(c.ref_grad)), at[1], at[2])

        a1 = jl_vs_jax_traj < 0.2 * jax_vs_f64_traj
        a2 = jl_vs_jax_g    < 0.5 * jax_vs_f64_g
        b1 = jl_vs_f64_traj < 2.0 * jax_vs_f64_traj
        b2 = jl_vs_f64_g    < 2.0 * jax_vs_f64_g
        ok = a1 && a2 && b1 && b2
        @printf("  closer-to-jax-than-jax-is-to-f64: traj %s grad %s | no-worse-vs-f64: traj %s grad %s  -> %s\n\n",
                a1 ? "ok" : "NO", a2 ? "ok" : "NO", b1 ? "ok" : "NO", b2 ? "ok" : "NO",
                ok ? "PASS" : "*** FAIL ***")
        allpass &= ok
    end
    println(allpass ? "ALL CONFIGS PASS" : "SOME CONFIGS FAILED")
    return allpass
end

main() || exit(1)
