# DifferLand/util/normalization.py, transcribed literally.
@inline _nor2par(p, mn, mx) = mn * (mx / mn)^p
@inline nor2par(x, mn, mx)  = _nor2par(atan(x) / Float32(pi) + 0.5f0, mn, mx)
@inline _par2nor(p, mn, mx) = log(p / mn) / log(mx / mn)
@inline par2nor(x, mn, mx)  = tan((_par2nor(x, mn, mx) - 0.5f0) * Float32(pi))
