"""Immutable per-site inputs. Met and target are stored transposed -- (18,T) and
(10,T) -- so one timestep is a contiguous column, which is the natural Julia
layout for a sequential scan (XLA picks its own layout on the JAX side)."""
struct Site
    T::Int
    train_end::Int
    ce_opt::Float32
    metT::Matrix{Float32}      # (18, T)
    tgtT::Matrix{Float32}      # (10, T)
    bnd::Bounds
    pmin::SVector{38,Float32}
    pmax::SVector{38,Float32}
    eoy::Vector{Float32}        # met row M_EOY
    eoy_denom::Float32          # sum(eoy) + 1
    total_precip::Float32       # sum over ALL timesteps of met[M_PREC, :]
    mean_ntemp::Float32         # mean over the training period of met[M_NTEMP, :]
end

function Site(metT::Matrix{Float32}, tgtT::Matrix{Float32}, train_end::Int,
              ce_opt::Real, bnd::Bounds)
    T = size(metT, 2)
    eoy = Float32[metT[M_EOY, t] for t in 1:T]
    Site(T, train_end, Float32(ce_opt), metT, tgtT, bnd,
         SVector{38,Float32}(bnd.parmin), SVector{38,Float32}(bnd.parmax),
         eoy, sum(eoy) + 1f0,
         sum(@view metT[M_PREC, :]),
         sum(@view metT[M_NTEMP, 1:train_end]) / Float32(train_end))
end

# ---- parameter vector layout -------------------------------------------------
# theta[1:30]  normalized physical parameters
# theta[31:38] normalized initial pools
# theta[39:  ] MLP weights/biases, layer by layer, weights first, in the same
#              flat order numpy writes a (n_in, n_out) array (C order).
nparams(::Default) = 38 + 4
nparams(::NNWhole) = 38 + (60 + 10) + (100 + 10) + (20 + 2)

@inline function unpack_mlp(::Default, θ)
    o = 38
    return ((Wt = SMatrix{1,1,Float32}(θ[o+1]), b = SVector{1,Float32}(θ[o+2])),
            (Wt = SMatrix{1,1,Float32}(θ[o+3]), b = SVector{1,Float32}(θ[o+4])))
end

@inline function unpack_mlp(::NNWhole, θ)
    o = 38
    W1 = SMatrix{10,6,Float32}(ntuple(i -> θ[o+i],       Val(60)))
    b1 = SVector{10,Float32}(  ntuple(i -> θ[o+60+i],    Val(10)))
    W2 = SMatrix{10,10,Float32}(ntuple(i -> θ[o+70+i],   Val(100)))
    b2 = SVector{10,Float32}(  ntuple(i -> θ[o+170+i],   Val(10)))
    W3 = SMatrix{2,10,Float32}(ntuple(i -> θ[o+180+i],   Val(20)))
    b3 = SVector{2,Float32}(   ntuple(i -> θ[o+200+i],   Val(2)))
    return ((Wt=W1,b=b1), (Wt=W2,b=b2), (Wt=W3,b=b3))
end
