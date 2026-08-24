# DALEC990.step -- literal transcription of DifferLand/model/DALEC990.py:31-238.
# `jnp.where` -> `ifelse` (both branches evaluated, so NaN and gradient
# semantics match). Statement order is preserved exactly; several statements
# overwrite a flux that a later statement reads, so reordering changes results.

@inline function gpp_and_et(::Default, mlp, paw_pool, lai, m,
                            ce, uWUE, boese_r, field_capacity, wilting_point_frac)
    wilting_point = field_capacity * wilting_point_frac
    beta = (paw_pool - wilting_point) / (field_capacity - wilting_point)
    beta = ifelse(beta <= 1f0, beta, 1f0)
    beta = ifelse(beta >= 0f0, beta, 0f0)
    gpp = ACM(m[M_LAT], m[M_DOY], m[M_TMAX], m[M_TMIN], lai, m[M_RAD], m[M_CA], ce) * beta
    ET  = gpp * sqrt(m[M_VPD]) / uWUE + m[M_RAD] * boese_r
    return gpp, ET, beta
end

@inline function gpp_and_et(::NNWhole, mlp, paw_pool, lai, m,
                            ce, uWUE, boese_r, field_capacity, wilting_point_frac)
    x = SVector{6,Float32}(m[M_NTEMP], lai / 8f0, m[M_NSOLAR],
                           paw_pool / 1500f0, m[M_NVPD], m[M_NCA])
    y = mlp_forward(mlp, x)
    gpp = max(0.01f0 * y[1], y[1])
    ET  = max(0.01f0 * y[2], y[2])
    return gpp, ET, -9999f0
end

@inline function step!(out, t, pools::SVector{8,Float32}, m, dp::SVector{30,Float32},
                       st::StressType, mlp, pmin::SVector{38,Float32},
                       pmax::SVector{38,Float32})
    time = m[M_TIME]; t_min = m[M_TMIN]; t_max = m[M_TMAX]
    precipitation = m[M_PREC]; delta_t = m[M_DT]

    labile_pool = pools[1]; foliar_pool = pools[2]; root_pool = pools[3]
    wood_pool   = pools[4]; litter_pool = pools[5]; som_pool  = pools[6]
    paw_pool    = pools[7]; puw_pool    = pools[8]

    decomposition_rate = dp[1]; f_auto = dp[2]; f_fol = dp[3]; f_root = dp[4]
    leaf_lifespan = dp[5]; tor_wood = dp[6]; tor_root = dp[7]
    tor_litter = dp[8]; tor_som = dp[9]; Q10 = dp[10]; ce = dp[11]
    Bday = dp[12]; f_lab = dp[13]; clab_release_period = dp[14]
    Fday = dp[15]; leaf_fall_period = dp[16]; LCMA = dp[17]; uWUE = dp[18]
    PAW_Qmax = dp[19]; field_capacity = dp[20]; wilting_point_frac = dp[21]
    lab_lifespan = dp[26]; moisture_factor = dp[27]; h2o_xfer = dp[28]
    PUW_Qmax = dp[29]; boese_r = dp[30]

    lai = foliar_pool / LCMA
    gpp, ET, beta = gpp_and_et(st, mlp, paw_pool, lai, m, ce, uWUE, boese_r,
                               field_capacity, wilting_point_frac)

    temperate = exp(Q10 * (0.5f0 * (t_max + t_min) - m[M_TMEAN])) *
                ((precipitation / m[M_MEANPREC] - 1f0) * moisture_factor + 1f0)

    respiration_auto  = f_auto * gpp
    leaf_production   = (gpp - respiration_auto) * f_fol
    labile_production = (gpp - respiration_auto - leaf_production) * f_lab
    root_production   = (gpp - respiration_auto - leaf_production - labile_production) * f_root
    wood_production   = gpp - respiration_auto - leaf_production - labile_production - root_production

    lff = leaf_fall_factor(time, leaf_lifespan, leaf_fall_period, Fday)
    lrf = lab_release_factor(time, lab_lifespan, clab_release_period, Bday)

    labile_release = labile_pool * (1f0 - (1f0 - lrf)^delta_t) / delta_t
    leaf_litter    = foliar_pool * (1f0 - (1f0 - lff)^delta_t) / delta_t
    wood_litter    = wood_pool   * (1f0 - (1f0 - tor_wood)^delta_t) / delta_t
    root_litter    = root_pool   * (1f0 - (1f0 - tor_root)^delta_t) / delta_t

    respiration_hetero_litter = litter_pool * (1f0 - (1f0 - temperate*tor_litter)^delta_t) / delta_t
    respiration_hetero_som    = som_pool    * (1f0 - (1f0 - temperate*tor_som)^delta_t) / delta_t
    litter_to_som             = litter_pool * (1f0 - (1f0 - temperate*decomposition_rate)^delta_t) / delta_t

    q_paw   = paw_pool^2 / PAW_Qmax / delta_t * (1f0 - h2o_xfer)
    paw2puw = q_paw * h2o_xfer / (1f0 - h2o_xfer)
    paw_focal_sel = paw_pool <= PAW_Qmax / 2f0
    q_paw   = ifelse(paw_focal_sel, q_paw,   (paw_pool - PAW_Qmax/4f0)/delta_t * (1f0 - h2o_xfer))
    paw2puw = ifelse(paw_focal_sel, paw2puw, (paw_pool - PAW_Qmax/4f0)/delta_t * h2o_xfer)

    q_puw = puw_pool^2 / PUW_Qmax / delta_t
    puw_focal_sel = puw_pool <= PUW_Qmax / 2f0
    q_puw = ifelse(puw_focal_sel, q_puw, (puw_pool - PUW_Qmax/4f0)/delta_t)

    # ---- pool updates: clamp to [parmin, parmax], back out the implied flux ----
    next_labile_pool = labile_pool + (labile_production - labile_release) * delta_t
    Clab_min_sel = next_labile_pool >= pmin[I_CLAB]
    next_labile_pool = ifelse(Clab_min_sel, next_labile_pool, pmin[I_CLAB])
    labile_release = ifelse(Clab_min_sel, labile_release,
                            labile_production - (next_labile_pool - labile_pool)/delta_t)
    Clab_max_sel = next_labile_pool <= pmax[I_CLAB]
    next_labile_pool = ifelse(Clab_max_sel, next_labile_pool, pmax[I_CLAB])
    labile_release = ifelse(Clab_max_sel, labile_release,
                            labile_production - (next_labile_pool - labile_pool)/delta_t)

    next_foliar_pool = foliar_pool + (leaf_production - leaf_litter + labile_release) * delta_t
    Cfol_min_sel = next_foliar_pool >= pmin[I_CFOL]
    next_foliar_pool = ifelse(Cfol_min_sel, next_foliar_pool, pmin[I_CFOL])
    # BUGCOMPAT cfol_min_leaf_litter: upstream false-branch is `(1-Cfol_min_sel)*parmin.Cfol`,
    # which for a scalar bool is just parmin.Cfol -- not the flux-balance expression
    # used in every sibling clamp. Reproduced verbatim.
    leaf_litter = ifelse(Cfol_min_sel, leaf_litter, pmin[I_CFOL])
    Cfol_max_sel = next_foliar_pool <= pmax[I_CFOL]
    next_foliar_pool = ifelse(Cfol_max_sel, next_foliar_pool, pmax[I_CFOL])
    leaf_litter = ifelse(Cfol_max_sel, leaf_litter,
                         leaf_production + labile_release - (next_foliar_pool - foliar_pool)/delta_t)

    next_root_pool = root_pool + (root_production - root_litter) * delta_t
    Croot_min_sel = next_root_pool >= pmin[I_CROOT]
    next_root_pool = ifelse(Croot_min_sel, next_root_pool, pmin[I_CROOT])
    root_litter = ifelse(Croot_min_sel, root_litter,
                         root_production - (next_root_pool - root_pool)/delta_t)
    Croot_max_sel = next_root_pool <= pmax[I_CROOT]
    next_root_pool = ifelse(Croot_max_sel, next_root_pool, pmax[I_CROOT])
    root_litter = ifelse(Croot_max_sel, root_litter,
                         root_production - (next_root_pool - root_pool)/delta_t)

    next_wood_pool = wood_pool + (wood_production - wood_litter) * delta_t
    Cwood_min_sel = next_wood_pool >= pmin[I_CWOOD]
    next_wood_pool = ifelse(Cwood_min_sel, next_wood_pool, pmin[I_CWOOD])
    wood_litter = ifelse(Cwood_min_sel, wood_litter,
                         wood_production - (next_wood_pool - wood_pool)/delta_t)
    Cwood_max_sel = next_wood_pool <= pmax[I_CWOOD]
    next_wood_pool = ifelse(Cwood_max_sel, next_wood_pool, pmax[I_CWOOD])
    wood_litter = ifelse(Cwood_max_sel, wood_litter,
                         wood_production - (next_wood_pool - wood_pool)/delta_t)

    next_litter_pool = litter_pool +
        (leaf_litter + root_litter - respiration_hetero_litter - litter_to_som) * delta_t
    Clitter_min_sel = next_litter_pool >= pmin[I_CLITTER]
    next_litter_pool = ifelse(Clitter_min_sel, next_litter_pool, pmin[I_CLITTER])
    litter_to_som = ifelse(Clitter_min_sel, litter_to_som,
        leaf_litter + root_litter - respiration_hetero_litter -
        (next_litter_pool - litter_pool)/delta_t)
    litter_to_som_sel = litter_to_som >= 0f0
    litter_to_som = ifelse(litter_to_som_sel, litter_to_som, 0f0)
    respiration_hetero_litter = ifelse(litter_to_som_sel, respiration_hetero_litter,
        leaf_litter + root_litter - (next_litter_pool - litter_pool)/delta_t)
    Clitter_max_sel = next_litter_pool <= pmax[I_CLITTER]
    next_litter_pool = ifelse(Clitter_max_sel, next_litter_pool, pmax[I_CLITTER])
    litter_to_som = ifelse(Clitter_max_sel, litter_to_som,
        leaf_litter + root_litter - respiration_hetero_litter -
        (next_litter_pool - litter_pool)/delta_t)

    next_som_pool = som_pool +
        (litter_to_som - respiration_hetero_som + wood_litter) * delta_t
    Csom_min_sel = next_som_pool >= pmin[I_CSOM]
    next_som_pool = ifelse(Csom_min_sel, next_som_pool, pmin[I_CSOM])
    respiration_hetero_som = ifelse(Csom_min_sel, respiration_hetero_som,
        litter_to_som + wood_litter - (next_som_pool - som_pool)/delta_t)
    Csom_max_sel = next_som_pool <= pmax[I_CSOM]
    next_som_pool = ifelse(Csom_max_sel, next_som_pool, pmax[I_CSOM])
    respiration_hetero_som = ifelse(Csom_max_sel, respiration_hetero_som,
        litter_to_som + wood_litter - (next_som_pool - som_pool)/delta_t)

    next_paw_pool = paw_pool + (-q_paw - paw2puw + precipitation - ET) * delta_t
    water_min_paw_sel = next_paw_pool >= pmin[I_PAW]
    next_paw_pool = ifelse(water_min_paw_sel, next_paw_pool, pmin[I_PAW])
    q_paw = ifelse(water_min_paw_sel, q_paw,
        precipitation - ET - (next_paw_pool - paw_pool)/delta_t * (1f0 - h2o_xfer))
    paw2puw = ifelse(water_min_paw_sel, paw2puw,
        precipitation - ET - (next_paw_pool - paw_pool)/delta_t * h2o_xfer)
    q_paw_sel = q_paw >= 0.0f0
    violation = max(-q_paw * 0.01f0, 0f0)
    ET      = ifelse(q_paw_sel, ET, precipitation - (next_paw_pool - paw_pool)/delta_t)
    q_paw   = ifelse(q_paw_sel, q_paw, 0f0)
    paw2puw = ifelse(q_paw_sel, paw2puw, 0f0)
    water_max_paw_sel = next_paw_pool <= pmax[I_PAW]
    next_paw_pool = ifelse(water_max_paw_sel, next_paw_pool, pmax[I_PAW])
    q_paw = ifelse(water_max_paw_sel, q_paw,
        precipitation - ET - (next_paw_pool - paw_pool)/delta_t * (1f0 - h2o_xfer))
    paw2puw = ifelse(water_max_paw_sel, paw2puw,
        precipitation - ET - (next_paw_pool - paw_pool)/delta_t * h2o_xfer)

    next_puw_pool = puw_pool + (paw2puw - q_puw) * delta_t
    next_puw_min_sel = next_puw_pool >= pmin[I_PUW]
    next_puw_pool = ifelse(next_puw_min_sel, next_puw_pool, pmin[I_PUW])
    q_puw = ifelse(next_puw_min_sel, q_puw, paw2puw - (next_puw_pool - puw_pool)/delta_t)
    next_puw_max_sel = next_puw_pool <= pmax[I_PUW]
    next_puw_pool = ifelse(next_puw_max_sel, next_puw_pool, pmax[I_PUW])
    # BUGCOMPAT puw_max_uses_min_sel: upstream gates this on next_puw_min_sel, not
    # next_puw_max_sel -- so it re-derives q_puw from the max-clamped pool only when
    # the *min* clamp was inactive. Reproduced verbatim.
    q_puw = ifelse(next_puw_min_sel, q_puw, paw2puw - (next_puw_pool - puw_pool)/delta_t)

    nee = -gpp + respiration_auto + respiration_hetero_litter + respiration_hetero_som

    @inbounds begin
        out[O_LAI, t] = lai;                       out[O_GPP, t] = gpp
        out[O_ET, t] = ET;                         out[O_TEMPERATE, t] = temperate
        out[O_RA, t] = respiration_auto;           out[O_LEAFPROD, t] = leaf_production
        out[O_LABPROD, t] = labile_production;     out[O_ROOTPROD, t] = root_production
        out[O_WOODPROD, t] = wood_production;      out[O_LFF, t] = lff
        out[O_LRF, t] = lrf;                       out[O_LABREL, t] = labile_release
        out[O_LEAFLIT, t] = leaf_litter;           out[O_WOODLIT, t] = wood_litter
        out[O_ROOTLIT, t] = root_litter;           out[O_RHLIT, t] = respiration_hetero_litter
        out[O_RHSOM, t] = respiration_hetero_som;  out[O_LIT2SOM, t] = litter_to_som
        out[O_QPAW, t] = q_paw;                    out[O_QPUW, t] = q_puw
        out[O_PAW2PUW, t] = paw2puw;               out[O_NEE, t] = nee
        out[O_PLAB, t] = next_labile_pool;         out[O_PFOL, t] = next_foliar_pool
        out[O_PROOT, t] = next_root_pool;          out[O_PWOOD, t] = next_wood_pool
        out[O_PLIT, t] = next_litter_pool;         out[O_PSOM, t] = next_som_pool
        out[O_PPAW, t] = next_paw_pool;            out[O_PPUW, t] = next_puw_pool
        out[O_BETA, t] = beta;                     out[O_VIOLATION, t] = violation
    end

    return SVector{8,Float32}(next_labile_pool, next_foliar_pool, next_root_pool,
                              next_wood_pool, next_litter_pool, next_som_pool,
                              next_paw_pool, next_puw_pool)
end
