"""forward!(out, theta, site, st) -- DALEC990.forward: unnormalize, then scan.

Writes the (32, T) flux/pool matrix in place. Mirrors `jax.lax.scan` over
`step`, except that JAX stacks the per-step outputs and we write them into a
caller-owned buffer (which is what lets the reverse pass avoid a tape copy).
"""
function forward!(out::AbstractMatrix{Float32}, θ::AbstractVector{Float32},
                  site::Site, st::StressType)
    b = site.bnd
    dp = SVector{30,Float32}(ntuple(i -> nor2par(θ[i], b.param_min[i], b.param_max[i]), Val(30)))
    p0 = SVector{8,Float32}(ntuple(i -> nor2par(θ[30+i], b.pool_min[i], b.pool_max[i]), Val(8)))
    mlp = unpack_mlp(st, θ)
    pmin = site.pmin; pmax = site.pmax; metT = site.metT
    pools = p0
    @inbounds for t in 1:site.T
        m = SVector{18,Float32}(ntuple(j -> metT[j, t], Val(18)))
        pools = step!(out, t, pools, m, dp, st, mlp, pmin, pmax)
    end
    return nothing
end

"""Unnormalized initial pools (`Pstart` in post_edc)."""
@inline function initial_pools(θ, site::Site)
    b = site.bnd
    SVector{8,Float32}(ntuple(i -> nor2par(θ[30+i], b.pool_min[i], b.pool_max[i]), Val(8)))
end

@inline function dalec_params(θ, site::Site)
    b = site.bnd
    SVector{30,Float32}(ntuple(i -> nor2par(θ[i], b.param_min[i], b.param_max[i]), Val(30)))
end
