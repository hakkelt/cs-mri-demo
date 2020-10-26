using Random, FFTW, Statistics, LinearAlgebra, EllipsisNotation, Wavelets, PaddedViews,
    AbstractOperators, OffsetArrays, PyCall, Distributed, ConfParser, JLD
include("functions.jl")

print("Reading data... ")
@load "data.jld"
imSize, imType, kSize, kType = size(image4D), eltype(image4D), size(kSpace), eltype(kSpace)
println("done")

@assert length(ARGS) == 1

println("Reading conf file: $(ARGS[1]).conf")
conf = ConfParse("$(ARGS[1]).conf")
parse_conf!(conf)

N = parse(Int, retrieve(conf, "N"))
undersampling_rate = parse(Float64, retrieve(conf, "undersampling_rate"))
fully_sampled_radius = parse(Float64, retrieve(conf, "fully_sampled_radius"))
λ₁ = parse(real(imType), retrieve(conf, "lambda_1"))
λ₂ = parse(real(imType), retrieve(conf, "lambda_2"))
λ₃ = parse(real(imType), retrieve(conf, "lambda_3"))
verbose = parse(Bool, retrieve(conf, "verbose"))
println("done")
#-----------------------------------------------

Random.seed!(1)

print("Generate mask... ")
pdf = generatePDF(floor(Int,imSize[1]/2) + 1, undersampling_rate, r = fully_sampled_radius)
pdf_2D = pdf * ones(imSize[3])'
mask = generate_mask(pdf_2D, 50, 0.01)
println("done")

print("Simulate measurement... ")
plan = plan_rfft(image4D, [1,2,3])
fft_temp = similar(plan * image4D)
inv_plan = plan_irfft(fft_temp, imSize[1], [1,2,3]);
normalizer = √(prod(imSize[1:3]) / 2)
Fᵤ = MyLinOp(imType, imSize, kType, kSize,
    (b, x) -> b .= 1/normalizer .* mask .* mul!(b, plan, x),
    (b, x) -> b .= normalizer .* mul!(b, inv_plan, fft_temp .= x))

y = Fᵤ * image4D
println("done")

image3D = sum(image4D, dims = 4)
image3D_norm = norm(image3D)
mse_temp = similar(image3D)
mse(img) = norm(sum!(mse_temp, img) .-= image3D) / image3D_norm

print("Zero-filling reconstruction... ")
image4D_zf = Fᵤ' * y
println("done")

print("Preparations... ")
norm₁(v) = norm(v, 1)
norm₂(x) = norm(x)
norm₂²(x) = abs(vec(x)' * vec(x))

wt = wavelet(WT.Daubechies{4}(), WT.Filter, WT.Periodic)
transformed_size = ((2 .^ (ceil.(Int, log2.(imSize[1:3]))))..., imSize[4])
temp_wavelet = similar(image3D, transformed_size)
oneTo(x) = 1:x
Ψ = MyLinOp(imType, imSize, transformed_size,
    (b, x) -> begin
        img = PaddedView(0, x, size(b))
        Threads.@threads for i in 1:imSize[4]
            @views dwt!(b[..,i], img[..,i], wt)
        end
        b
    end,
    (b, x) -> begin
        Threads.@threads for i in 1:imSize[4]
            @views idwt!(temp_wavelet[..,i], x[..,i], wt)
        end
        b .= @view temp_wavelet[oneTo.(imSize)...]
    end)

TV = MyLinOp(imType, imSize, (imSize[1:3]..., 3, imSize[end]),
    (∇, image) -> begin
        Threads.@threads for i in 1:imSize[end]
            for d in 1:ndims(image)-1
                @views _finite_differences!(∇[..,d, i], image[..,i], d)
            end
        end
        ∇
    end,
    (∇, image) -> begin
        Threads.@threads for i in 1:imSize[end]
            for d in 1:ndims(∇)-1
                @views _second_order_finite_differences!(∇[..,i], image[..,d, i], d, d == 1) 
            end
        end
        ∇
    end)

prox_tv = pyimport("prox_tv")
temp₁, temp₂, temp₃, temp₄ = Fᵤ * image4D, Ψ * image4D, TV * image4D, similar(image4D)
f_pogm(x) = norm₂²(mul!(temp₁, Fᵤ, x) .-= y)
∇f_pogm!(b, x) = begin
    #2 * Fᵤ' * (Fᵤ * x - y)
    mul!(temp₁, Fᵤ, x)
    temp₁ .-= y
    mul!(b, Fᵤ', temp₁)
    b .*= 2
end
g_pogm(x) = (λ₁ != 0 ? λ₁ * norm(mul!(temp₂, Ψ, x), 1) : 0) +
            (λ₂ != 0 ? λ₂ * norm(mul!(temp₃, TV, x), 1) : 0) +
            (λ₃ != 0 ? λ₃ * sum(svdvals(reshape(x, :, imSize[end]))) : 0)
prox_g_pogm!(b, x, γ) = begin
    if λ₂ != 0
        for i in 1:imSize[end]
            @views b[..,i] .= prox_tv.tvgen(x[..,i], (γ*λ₂, γ*λ₂, γ*λ₂), (1, 2, 3), (1, 1, 1))
        end
    end
    λ₁ != 0 && mul!(b, Ψ', Λ!(mul!(temp₂, Ψ, b), γ * λ₁))
    λ₃ != 0 && SVT!(reshape(b, :, imSize[end]), λ₃)
    b
end
println("done")

println("Reconstruction...")
@time image4D_pogm, f_vec_pogm, NMSE_vec_pogm = POGM(image4D_zf, f_pogm, ∇f_pogm!, g_pogm, prox_g_pogm!,
    N = N, L = 1, verbose = verbose)
println("done")

@save "$(ARGS[1]).jld" image4D_pogm f_vec_pogm NMSE_vec_pogm


