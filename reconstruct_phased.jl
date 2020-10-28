using Random, FFTW, Statistics, LinearAlgebra, EllipsisNotation, Wavelets, PaddedViews,
    AbstractOperators, OffsetArrays, PyCall, Distributed, JLD, ConfParser
include("functions_phased.jl")

print("Reading data... ")
@load "true_data.jld"
imSize, imType, kSize, kType = size(image4D), eltype(image4D), size(kSpace), eltype(kSpace)
println("done")

@assert length(ARGS) == 1

print("Reading conf file: $(ARGS[1]).conf...")
conf = ConfParse("$(ARGS[1]).conf")
parse_conf!(conf)

N = parse(Int, retrieve(conf, "N"))
undersampling_rate = parse(Float64, retrieve(conf, "undersampling_rate"))
fully_sampled_radius = parse(Float64, retrieve(conf, "fully_sampled_radius"))
λ₁ = parse(real(imType), retrieve(conf, "lambda_1"))
λ₂ = parse(real(imType), retrieve(conf, "lambda_2"))
λ₃ = parse(real(imType), retrieve(conf, "lambda_3"))
randomize_4D = parse(Bool, retrieve(conf, "randomize_4D"))
restart = parse(Bool, retrieve(conf, "restart"))
verbose = parse(Bool, retrieve(conf, "verbose"))
println("done")
#-----------------------------------------------

Random.seed!(1)

print("Generate mask... ")
pdf₁ = generatePDF(imSize[1], √undersampling_rate, r = fully_sampled_radius)
pdf₂ = generatePDF(imSize[3], √undersampling_rate, r = fully_sampled_radius)
pdf_2D = pdf₁ * pdf₂'
mask = generate_mask(pdf_2D, 200, 0.005, randomize_4D)
println("done")

print("Simulate measurement... ")
plan = plan_fft(image4D, [1,2,3])
kSpace_temp = similar(plan * image4D)
image_temp = similar(image4D)
inv_plan = plan_ifft(kSpace_temp, [1,2,3]);
normalizer = √(prod(imSize[1:3]))
Fᵤ = MyLinOp(imType, imSize, kType, kSize,
    (b, x) -> b .= 1/normalizer .* mask .* mul!(b, plan, ifftshift!(kSpace_temp, x, 1)),
    (b, x) -> fftshift!(b, image_temp .= normalizer .* mul!(b, inv_plan, kSpace_temp .= x), 1))

y = Fᵤ * image4D
println("done")

image3D = sum(abs.(image4D), dims = 4)
image3D_norm = norm(image3D)
mse_temp = similar(image3D)
mse(img) = norm(sum!(mse_temp, abs.(img)) .-= image3D) / image3D_norm

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
Ψ = MyLinOp(real(imType), imSize, transformed_size,
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

TV = MyLinOp(real(imType), imSize, (imSize[1:3]..., 3, imSize[end]),
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
temp₁, temp₂, temp₃ = Fᵤ * image4D, Ψ * abs.(image4D), TV * abs.(image4D)
temp₄, phase = [similar(image4D, real(imType)) for _ in 1:2]
f_pogm(x) = norm₂²(mul!(temp₁, Fᵤ, x) .-= y)
∇f_pogm!(b, x) = begin
    #2 * Fᵤ' * (Fᵤ * x - y)
    mul!(temp₁, Fᵤ, x)
    temp₁ .-= y
    mul!(b, Fᵤ', temp₁)
    b .*= 2
end
g_pogm(x) = (λ₁ != 0 ? λ₁ * norm(mul!(temp₂, Ψ, temp₄ .= abs.(x)), 1) : 0) +
                (λ₂ != 0 ? λ₂ * norm(mul!(temp₃, TV, temp₄ .= abs.(x)), 1) : 0) +
                (λ₃ != 0 ? λ₃ * sum(svdvals(reshape(temp₄ .= abs.(x), :, imSize[end]))) : 0)
prox_g_pogm!(b, x, γ) = begin
    phase .= angle.(x)
    temp₄ .= abs.(x)
    if λ₁ != 0
        for i in 1:imSize[end]
            @views temp₄[..,i] .= prox_tv.tvgen(temp₄[..,i], (γ*λ₂, γ*λ₂, γ*λ₂), (1, 2, 3), (1, 1, 1))
        end
    end
    λ₂ != 0 && mul!(temp₄, Ψ', Λ!(mul!(temp₂, Ψ, temp₄), γ * λ₁))
    λ₃ != 0 && SVT!(reshape(temp₄, :, imSize[end]), λ₃)
    b .= temp₄ .* exp.(1im .* phase)
end
println("done")

println("Reconstruction...")
@time image4D_pogm, f_vec_pogm, NMSE_vec_pogm = POGM(image4D_zf, f_pogm, ∇f_pogm!, g_pogm, prox_g_pogm!,
    N = N, L = 1, fname = "$(ARGS[1]).jld", restart = restart, verbose = verbose)
println("done")
