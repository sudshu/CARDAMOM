"""optax.adam, transcribed: scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0)
followed by scale(-lr). Bias correction uses the 1-based step count, matching
optax's `count` after its increment."""
mutable struct Adam
    lr::Float32; b1::Float32; b2::Float32; eps::Float32
    m::Vector{Float32}; v::Vector{Float32}; t::Int
end
Adam(n::Int; lr = 5f-4, b1 = 0.9f0, b2 = 0.999f0, eps = 1f-8) =
    Adam(lr, b1, b2, eps, zeros(Float32, n), zeros(Float32, n), 0)

function adam_step!(θ::Vector{Float32}, g::Vector{Float32}, s::Adam)
    s.t += 1
    bc1 = 1f0 - s.b1^s.t
    bc2 = 1f0 - s.b2^s.t
    @inbounds @simd for i in eachindex(θ)
        s.m[i] = s.b1 * s.m[i] + (1f0 - s.b1) * g[i]
        s.v[i] = s.b2 * s.v[i] + (1f0 - s.b2) * g[i] * g[i]
        θ[i] -= s.lr * (s.m[i] / bc1) / (sqrt(s.v[i] / bc2) + s.eps)
    end
    return nothing
end

"""One calibration iteration: value_and_grad, then the Adam update."""
function calib_step!(θ, g, out, dout, site::Site, st::StressType, k::Float32, s::Adam)
    fill!(g, 0f0); fill!(dout, 0f0)
    _, L = Enzyme.autodiff(set_runtime_activity(ReverseWithPrimal), compute_loss!,
                           Duplicated(out, dout), Duplicated(θ, g),
                           Const(site), Const(st), Const(k))
    adam_step!(θ, g, s)
    return L
end

"""Loss and gradient in one reverse sweep (the quantity Adam consumes)."""
function value_and_grad!(g, out, dout, θ, site::Site, st::StressType, k::Float32)
    fill!(g, 0f0); fill!(dout, 0f0)
    _, L = Enzyme.autodiff(set_runtime_activity(ReverseWithPrimal), compute_loss!,
                           Duplicated(out, dout), Duplicated(θ, g),
                           Const(site), Const(st), Const(k))
    return L
end
