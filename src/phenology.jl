# DifferLand/model/auxi/phenology.py, transcribed literally (op order preserved).
@inline function offset(L, w)
    p1 =  0.000023599784710f0; p2 =  0.000332730053021f0
    p3 =  0.000901865258885f0; p4 = -0.005437736864888f0
    p5 = -0.020836027517787f0; p6 =  0.126972018064287f0
    p7 = -0.188459767342504f0
    lf = log(L - 1f0)
    os = p1*lf^6 + p2*lf^5 + p3*lf^4 + p4*lf^3 + p5*lf^2 + p6*lf + p7
    return os * w
end

@inline function lab_release_factor(t, lab_lifespan, clab_release_period, Bday)
    fl  = (log(lab_lifespan) - log(lab_lifespan - 1f0)) * 0.5f0
    wl  = clab_release_period * sqrt(2f0) / 2f0
    osl = offset(lab_lifespan, wl)
    sf  = 365.25f0 / Float32(pi)
    return (2f0 / sqrt(Float32(pi))) * (fl / wl) *
           exp(-(sin((t - Bday + osl) / sf) * sf / wl)^2)
end

@inline function leaf_fall_factor(t, leaf_lifespan, leaf_fall_period, Fday)
    ff  = (log(leaf_lifespan) - log(leaf_lifespan - 1f0)) * 0.5f0
    wf  = leaf_fall_period * sqrt(2f0) / 2f0
    osf = offset(leaf_lifespan, wf)
    sf  = 365.25f0 / Float32(pi)
    return (2f0 / sqrt(Float32(pi))) * (ff / wf) *
           exp(-(sin((t - Fday + osf) / sf) * sf / wf)^2)
end
