# DALEC990 parameter/pool index maps and output-column layout.
# Transcribed from DifferLand/model/DALEC_990_parinfo.py; the 1-based Julia
# indices are the python 0-based ones plus one.

# parmin_arr / parmax_arr slots that make up the 30 trainable physical params
const PARAM_IDX = Int32[1:17...,  24, 25, 26, 27,  29, 30, 31, 32, 33, 34, 35, 36,  38]
# ... and the 8 trainable initial pools
const POOL_IDX  = Int32[18, 19, 20, 21, 22, 23, 28, 37]

# named slots into the full 38-element bound arrays (used for the pool clamps)
const I_CLAB, I_CFOL, I_CROOT, I_CWOOD = 18, 19, 20, 21
const I_CLITTER, I_CSOM, I_PAW, I_PUW  = 22, 23, 28, 37

# dalec990_pfn: output-matrix columns (1-based)
const O_LAI, O_GPP, O_ET, O_TEMPERATE = 1, 2, 3, 4
const O_RA, O_LEAFPROD, O_LABPROD, O_ROOTPROD, O_WOODPROD = 5, 6, 7, 8, 9
const O_LFF, O_LRF, O_LABREL = 10, 11, 12
const O_LEAFLIT, O_WOODLIT, O_ROOTLIT = 13, 14, 15
const O_RHLIT, O_RHSOM, O_LIT2SOM = 16, 17, 18
const O_QPAW, O_QPUW, O_PAW2PUW, O_NEE = 19, 20, 21, 22
const O_PLAB, O_PFOL, O_PROOT, O_PWOOD = 23, 24, 25, 26
const O_PLIT, O_PSOM, O_PPAW, O_PPUW = 27, 28, 29, 30
const O_BETA, O_VIOLATION = 31, 32
const NOUT = 32
const N_FLUX = 22          # pfn.next_labile_pool: FLUXES = out[1:22, :]
const POOL_RANGE = O_PLAB:O_PPAW    # 7 pools (labile..paw), excludes puw

# met-matrix rows (we store met transposed: (18, T), one contiguous column/step)
const M_TIME, M_TMIN, M_TMAX, M_RAD, M_CA, M_DOY = 1, 2, 3, 4, 5, 6
const M_BURNED, M_VPD, M_PREC, M_LAT, M_DT = 7, 8, 9, 10, 11
const M_TMEAN, M_MEANPREC = 12, 13
const M_NTEMP, M_NSOLAR, M_NVPD, M_NCA, M_EOY = 14, 15, 16, 17, 18

struct Bounds
    parmin::Vector{Float32}      # 38
    parmax::Vector{Float32}      # 38
    param_min::Vector{Float32}   # 30
    param_max::Vector{Float32}
    pool_min::Vector{Float32}    # 8
    pool_max::Vector{Float32}
end

function Bounds(parmin::Vector{Float32}, parmax::Vector{Float32})
    Bounds(parmin, parmax,
           parmin[PARAM_IDX], parmax[PARAM_IDX],
           parmin[POOL_IDX],  parmax[POOL_IDX])
end
