# DifferLand/model/auxi/ACM.py, transcribed literally including the two
# no-op/quirk lines in the day-length clamp (see BUGCOMPAT notes in README).
@inline function ACM(lat, doy, t_max, t_min, lai, rad, ca, ce)
    d1 = 0.0156935f0; θ = 4.22273f0; k = 208.868f0; d2 = 0.0453194f0
    b2 = 0.37836f0;   c1 = 7.19298f0; a2 = 0.011136f0; c2 = 2.1001f0
    eb1T = 0.789798f0; ψ_d = -2f0; H = 1f0
    PI = Float32(pi)

    gc = abs(ψ_d)^eb1T / (b2 * H + 0.5f0 * (t_max - t_min))
    p  = lai * 1f0 * ce * exp(a2 * t_max) / gc
    q  = θ - k
    ci = 0.5f0 * (ca + q - p + sqrt((ca + q - p)^2 - 4f0 * (ca * q - p * θ)))
    e0 = c1 * (lai^2) / (c2 + lai^2)

    dec  = -23.4f0 * cos((360f0 * (doy + 10f0) / 365f0) * PI / 180f0) * PI / 180f0
    mult = tan(lat * PI / 180f0) * tan(dec)
    mult_valid = (mult < 1f0) * (mult > -1f0)
    mult_temp  = mult * mult_valid
    dayl = 24f0 * acos(-mult_temp) / PI
    mult_geq_one_sel = (mult < 1f0)
    dayl = dayl * mult_geq_one_sel + (1f0 - mult_geq_one_sel) * 24f0
    mult_leq_minus_one_sel = (mult > -1f0)
    # BUGCOMPAT acm_dayl_zero: second term is `(1-mult_geq_one_sel)*0`, i.e. always
    # zero, so this line only masks dayl by (mult > -1). Reproduced verbatim.
    dayl = dayl * mult_leq_minus_one_sel + (1f0 - mult_geq_one_sel) * 0f0

    pd = gc * (ca - ci)
    pei = e0 * rad * pd / (e0 * rad + pd)
    return pei * (d1 * dayl + d2)
end
