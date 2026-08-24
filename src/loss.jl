# DifferLand/optimization/loss_functions.py + DALEC990.{pre_edc,post_edc,compute_loss}

@inline negative_log_sigmoid(a, b, k) = -log(1f0 / (1f0 + exp(-k * (a - b))))

"""compute_nnse(observed, modeled, mask). `md(i)` supplies the modeled value so
the RECO target can be NEE+GPP without materializing a temporary."""
@inline function nnse(obs, msk, md::F, n::Int) where {F}
    numerator = sum(i -> (obs[i] - md(i))^2 * msk[i], 1:n)
    msum = sum(i -> msk[i], 1:n)
    obs_mean = sum(i -> obs[i] * msk[i], 1:n) / msum
    denominator = sum(i -> (obs[i] - obs_mean)^2 * msk[i], 1:n)
    nse = 1f0 - numerator / denominator
    return 1f0 / (2f0 - nse)
end

function pre_edc(dp::SVector{30,Float32}, mean_temp::Float32, k::Float32)
    a_auto = dp[2]
    a_fol  = (1f0 - a_auto) * dp[3]
    a_lab  = (1f0 - a_auto - a_fol) * dp[13]
    a_root = (1f0 - a_auto - a_fol - a_lab) * dp[4]

    edc1 = negative_log_sigmoid(dp[8], dp[9], 100000f0 * k)
    edc2 = negative_log_sigmoid(dp[1], dp[9], 10000f0 * k)
    edc3 = negative_log_sigmoid(1f0 / (dp[5] * 365.25f0), dp[6], 200000f0 * k)
    edc4 = negative_log_sigmoid(dp[7], dp[9] * exp(dp[10] * mean_temp), 10f0 * k)
    edc5 = negative_log_sigmoid(5f0 * a_root, a_fol + a_lab, 100f0 * k) +
           negative_log_sigmoid(5f0 * (a_fol + a_lab), a_root, 100f0 * k)
    return edc1 + edc2 + edc3 + edc4 + edc5
end

function post_edc(out::AbstractMatrix{Float32}, Pstart::SVector{8,Float32},
                  site::Site, k::Float32)
    T = site.T; eoy = site.eoy
    # MPOOLS_Jan: end-of-year pool means, seeded with the initial pools
    MP = SVector{7,Float32}(ntuple(
        j -> (Pstart[j] + sum(t -> out[POOL_RANGE[j], t] * eoy[t], 1:T)) / site.eoy_denom,
        Val(7)))
    # FTOTAL over all 22 flux columns (as upstream; the reduce is not sliced first)
    FT = SVector{N_FLUX,Float32}(ntuple(j -> sum(t -> out[j, t], 1:T), Val(N_FLUX)))

    Fin = SVector{7,Float32}(
        FT[O_LABPROD],
        FT[O_LEAFPROD] + FT[O_LABREL],
        FT[O_ROOTPROD],
        FT[O_WOODPROD],
        FT[O_LEAFLIT] + FT[O_ROOTLIT],
        FT[O_WOODLIT] + FT[O_LIT2SOM],
        site.total_precip)
    Fout = SVector{7,Float32}(
        FT[O_LABREL],
        FT[O_LEAFLIT],
        FT[O_ROOTLIT],
        FT[O_WOODLIT],
        FT[O_RHLIT] + FT[O_LIT2SOM],
        FT[O_RHSOM],
        FT[O_ET] + FT[O_QPAW] + FT[O_PAW2PUW])

    Rm = Fin ./ Fout
    Rs = Rm .* MP ./ SVector{7,Float32}(ntuple(j -> Pstart[j], Val(7)))

    EQF = 2f0
    edc = 0f0
    # upstream loops i in range(6): the 7th (water) ratio is deliberately unused here
    for i in 1:6
        edc += negative_log_sigmoid(log(EQF), abs(log(Rs[i])), 3f0 * k)
        edc += negative_log_sigmoid(0.1f0,   abs(Rs[i] - Rm[i]), 3f0 * k)
    end
    edc += sum(t -> out[O_VIOLATION, t], 1:T)
    return edc
end

"""compute_loss -- the quantity Adam differentiates. `out` is a caller-owned
(32, T) scratch buffer."""
function compute_loss!(out::AbstractMatrix{Float32}, θ::AbstractVector{Float32},
                       site::Site, st::StressType, k::Float32)
    forward!(out, θ, site, st)
    n = site.train_end
    tgt = site.tgtT

    gpp_loss = nnse((@view tgt[1, :]), (@view tgt[2, :]), i -> out[O_GPP, i], n)
    # reco=True: modeled RECO is NEE + GPP
    reco_loss = nnse((@view tgt[9, :]), (@view tgt[10, :]),
                     i -> out[O_NEE, i] + out[O_GPP, i], n)
    et_loss  = nnse((@view tgt[5, :]), (@view tgt[6, :]), i -> out[O_ET, i],  n)
    lai_loss = nnse((@view tgt[7, :]), (@view tgt[8, :]), i -> out[O_LAI, i], n)

    ce = nor2par(θ[11], 5f0, 50f0)
    ce_loss = ifelse(site.ce_opt >= 5f0, 0.1f0 * (ce - site.ce_opt)^2, 0f0)

    dp = dalec_params(θ, site)
    edc_loss = pre_edc(dp, site.mean_ntemp, k)
    edc_loss += post_edc(out, initial_pools(θ, site), site, k)

    return -gpp_loss - reco_loss - et_loss - lai_loss + ce_loss + edc_loss
end
