"""
DifferLand.jl -- Julia port of the differentiable DALEC990 hybrid land model
(Fang & Gentine, DifferLand v1.1), transcribed from the JAX implementation at
/export/data1/spandey/DifferLand/DifferLand_v1.1/DifferLand.

float32 throughout, matching the JAX original (which never enables x64).
Reverse-mode gradients come from Enzyme over the plain sequential loop.
"""
module DifferLand

using StaticArrays
using LinearAlgebra
using JSON
using Enzyme

export Site, Bounds, Default, NNWhole, forward!, compute_loss!, nparams,
       nor2par, par2nor, load_fixture, FixtureConfig,
       Adam, adam_step!, calib_step!, value_and_grad!

include("parinfo.jl")
include("stresstypes.jl")
include("normalization.jl")
include("phenology.jl")
include("acm.jl")
include("mlp.jl")
include("site.jl")
include("step.jl")
include("forward.jl")
include("loss.jl")
include("adam.jl")
include("fixtures.jl")

end # module
