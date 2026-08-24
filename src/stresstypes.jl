"""Water-stress configuration, dispatched at compile time (upstream selects with
a runtime string compare inside a jitted step)."""
abstract type StressType end
struct Default <: StressType end     # config 2: beta-JS
struct NNWhole <: StressType end     # config 5: GPP&ET(NN)_MET+LAI
