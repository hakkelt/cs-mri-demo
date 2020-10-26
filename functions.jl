
function generatePDF(dataLength, uf; distType = 2, r = 0.1, max_p = 50, disp = false)
    pdf = nothing
    
    for p = 0:max_p

        fs_end = floor(Int, r * dataLength)

        pdf = ones(dataLength)
        distance_map = abs.(range(0, 1, length = dataLength))
        pdf[fs_end+1:end] = (1 .- distance_map[fs_end+1:end]).^p
        
        # shift the exponential part to fullfil condition 4
        diff = uf * dataLength - sum(pdf)
        diff < 0 && continue
        pdf[fs_end+1:end] .+= diff / (dataLength - fs_end)
        all(x -> 0 <= x <= 1, pdf) && break
    end
    
    (pdf isa Nothing || any(x -> x < 0 || 1 < x, pdf)) && error("infeasible with given parameters")
    
    collect(pdf)
end

function generate_mask(pdf_2D, iter, tol; max_inner_iter = 100)
    function generate_2D_mask(pdf)
        mask = nothing
        for i in 1:max_inner_iter
            mask = [rand() < e for e in pdf]
            diff = abs(sum(mask) - sum(pdf))
            diff < tol*length(mask) && break
            i == max_inner_iter && error("infeasible with given parameters")
        end
        return mask
    end
    
    function extend_to_3D(mask)
        permutedims(repeat(mask, 1, 1, imSize[2]), [1,3,2])
    end
    
    masks = [generate_2D_mask(pdf_2D) for i = 1:iter]
    F⁻¹ₘₐₛₖₛ = [ifft(mask./pdf_2D) for mask in masks]                    # Inverse Fourier transform
    TPSFₘₐₓ = [maximum(abs.(F⁻¹ₘₐₛₖ[2:end])) for F⁻¹ₘₐₛₖ in F⁻¹ₘₐₛₖₛ] # height of larges sidelobe
    
    top_list = sort(TPSFₘₐₓ)[1:imSize[4]]
    selected_masks = [findfirst(isequal(elem), TPSFₘₐₓ) for elem in top_list]
    cat([extend_to_3D(masks[i]) for i in selected_masks]..., dims = 4) .== 1
end

function _finite_differences!(Δx, x, d)
    last = size(Δx, d)
    𝟘 = zero(eltype(x))
    for i in axes(Δx, d)
        if i != last
            selectdim(Δx, d, i) .= selectdim(x, d, i) .- selectdim(x, d, i+1)
        else
            selectdim(Δx, d, i) .= 𝟘
        end
    end
end

function _second_order_finite_differences!(Δx, x, d, overwrite)
    last = size(Δx, d)
    𝟘, 𝟙 = zero(eltype(x)), one(eltype(x))
    for i in axes(Δx, d)
        if overwrite
            if i == 1
                selectdim(Δx, d, i) .= -𝟙 .* selectdim(x, d, i)
            elseif i != last
                selectdim(Δx, d, i) .= selectdim(x, d, i) .- selectdim(x, d, i-1)
            else
                selectdim(Δx, d, i) .= 𝟘
            end
        else
            if i == 1
                selectdim(Δx, d, i) .-= selectdim(x, d, i)
            elseif i != last
                selectdim(Δx, d, i) .+= selectdim(x, d, i) .- selectdim(x, d, i-1)
            end
        end
    end
    Δx
end

smoothed_change(x) = begin
    mean(x[1:10]) - mean(x[10:20])
end

function POGM(x₀, f, ∇f!, g, prox_g!; N = 10, L = 1, restart = true, verbose = false)
    
    dType = eltype(x₀)
    θₖ₋₁ = γₖ₋₁ = one(real(dType))
    xₖ, xₖ₋₁, yₖ₋₁, yₖ, zₖ₋₁, zₖ, temp = [copy(x₀) for _ in 1:7]
    t = convert(real(dType), 1/2L)
    
    f_vec, g_vec, NMSE_vec = [OffsetVector{real(dType)}(undef, 0:N) for _ in 1:3]
    f_vec[0], g_vec[0], NMSE_vec[0] = f(xₖ₋₁), g(xₖ₋₁), mse(xₖ₋₁)
    if verbose
        println("k: 0, consistency: $(f_vec[0]), regularization: $(g_vec[0]), NMSE: $(NMSE_vec[0])")
    end
    
    for k in 1:N
        yₖ .= xₖ₋₁ .- t .* ∇f!(temp, xₖ₋₁)
        θₖ = (1 + √(1 + (k < N ? 4 : 8)*θₖ₋₁^2))/2
        @. zₖ = yₖ + (θₖ₋₁-1)/θₖ * (yₖ - yₖ₋₁) + θₖ₋₁/θₖ * (yₖ - xₖ₋₁) +
                t * (θₖ₋₁-1)/(γₖ₋₁*θₖ) * (zₖ₋₁ - xₖ₋₁)
        γₖ = t*(2θₖ₋₁ + θₖ₋₁ - 1)/θₖ
        prox_g!(xₖ, zₖ, 1)
        
        f_vec[k], g_vec[k], NMSE_vec[k] = f(xₖ), g(xₖ), mse(xₖ)
        if verbose
            println("k: $k, consistency: $(f_vec[k]), regularization: $(g_vec[k]), NMSE: $(NMSE_vec[k])")
        end
        
        if (k > 10 && NMSE_vec[k] > NMSE_vec[0]) || f_vec[k] > f_vec[0] * 100 || g_vec[k] > g_vec[0] * 100
            verbose && println("   Diverged!")
            break
        end
        
        if k > 20 && smoothed_change(NMSE_vec[k-20:k]) ./ NMSE_vec[0] < 1e-6
            verbose && println("   Converged!")
            break
        end
        
        if restart && f_vec[k] + g_vec[k] > f_vec[k - 1] + g_vec[k - 1]
            θₖ₋₁ = θₖ = one(real(dType))
            verbose && println("   Restarted!")
        end
        
        xₖ₋₁, yₖ₋₁, zₖ₋₁, xₖ, yₖ, zₖ = xₖ, yₖ, zₖ, xₖ₋₁, yₖ₋₁, zₖ₋₁
        θₖ₋₁, γₖ₋₁ = θₖ, γₖ
    end
    
    xₖ, f_vec .+ g_vec, NMSE_vec
end

pos(x) = x < 0 ? zero(x) : x
Λ!(v, p) = @. v = sign(v) * pos(abs(v) - p)
SVT!(A, p) = begin
    F = svd!(A)
    A .= F.U * Diagonal(Λ!(F.S, p)) * F.Vt
end
