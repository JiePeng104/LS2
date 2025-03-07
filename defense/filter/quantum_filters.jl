include("dkk17.jl")
using NPZ

function rcov_quantum_filter(reps, eps, k, α=4, τ=0.1; limit1=2, limit2=1.5)
    d, n = size(reps)
    reps_pca, U = pca(reps, k)
    if k == 1
        reps_estimated_white = reps_pca
        Σ′ = ones(1, 1)
    else
        selected = cov_estimation_iterate(reps_pca, eps/n, τ, nothing, limit=round(Int, limit1*eps))
        reps_pca_selected = reps_pca[:, selected]
        Σ = cov(reps_pca_selected', corrected=false)
#         reps_estimated_white = Σ^(-1/2)*reps_pca
        reps_estimated_white = sqrt(Hermitian(Σ))\reps_pca
        Σ′ = cov(reps_estimated_white')
    end
    M = k > 1 ? exp(α*(Σ′- I)/(opnorm(Σ′) - 1)) : ones(1, 1)
    M /= tr(M)
    estimated_poison_ind = k_lowest_ind(
        -[x'M*x for x in eachcol(reps_estimated_white)],
        round(Int, limit2*eps)
    )
    return .! estimated_poison_ind
end


function rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator, α=4, τ=0.1; limit1=2, limit2=1.5)
    reps_pca, U = pca(reps, 100)
    best_opnorm, best_selected, best_k = -Inf, nothing, nothing
    for k in round.(Int, range(1, sqrt(100), length=10) .^ 2)

        selected = rcov_quantum_filter(reps, eps, k, α, τ; limit1=limit1, limit2=limit2)
        Σ = cov(reps_pca[:, selected]')
#         Σ′ = cov((Σ^(-1/2)*reps_pca)')
        Σ′ = cov((sqrt(Hermitian(Σ))\reps_pca)')
        M = exp(α*(Σ′- I)/(opnorm(Σ′) - 1))
        M /= tr(M)
        op = tr(Σ′ * M) / tr(M)
        poison_removed = sum((.! selected)[end-eps+1:end])
        @show k, op, poison_removed
        if op > best_opnorm
            best_opnorm, best_selected, best_k = op, selected, k
        end

        r = eps*1.5
        poi_list = .! selected
        acc_poi = 0
        miss_poi = 0
        for i in 1:length(poi_indicator)
            if poi_indicator[i]==1
                if poi_list[i]
                    acc_poi += 1
                else
                    miss_poi += 1
                end
            end
        end
        ratio = acc_poi / r
        @printf(log_file, "quantum---PCA-K: %d, opnorm: %d, acc_poi: %d, miss_poi: %d, ratio: %f\n", k, op, acc_poi, miss_poi, ratio)
    end
    @show best_k, best_opnorm
    @printf(log_file, "quantum---best_k: %d, best_opnorm: %d\n", best_k, best_opnorm)
    return best_selected
end

function pcaall_rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator, α=4, τ=0.1; limit1=2, limit2=1.5)
    reps_pca, U = pca(reps, 100)
    best_opnorm, best_selected, best_k = -Inf, nothing, nothing
    for k in round.(Int, range(1, 100, length=100))

        selected = rcov_quantum_filter(reps, eps, k, α, τ; limit1=limit1, limit2=limit2)
        Σ = cov(reps_pca[:, selected]')
#         Σ′ = cov((Σ^(-1/2)*reps_pca)')
        Σ′ = cov((sqrt(Hermitian(Σ))\reps_pca)')
        M = exp(α*(Σ′- I)/(opnorm(Σ′) - 1))
        M /= tr(M)
        op = tr(Σ′ * M) / tr(M)
        poison_removed = sum((.! selected)[end-eps+1:end])
        @show k, op, poison_removed
        if op > best_opnorm
            best_opnorm, best_selected, best_k = op, selected, k
        end

        r = eps*1.5
        poi_list = .! selected
        acc_poi = 0
        miss_poi = 0
        for i in 1:length(poi_indicator)
            if poi_indicator[i]==1
                if poi_list[i]
                    acc_poi += 1
                else
                    miss_poi += 1
                end
            end
        end
        ratio = acc_poi / r
        @printf(log_file, "quantum---PCA-K: %d, opnorm: %d, acc_poi: %d, miss_poi: %d, ratio: %f\n", k, op, acc_poi, miss_poi, ratio)
    end
    @show best_k, best_opnorm
    @printf(log_file, "quantum---best_k: %d, best_opnorm: %d\n", best_k, best_opnorm)
    return best_selected
end

function tsne_rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator, α=4, τ=0.1; limit1=2, limit2=1.5)
    reps_pca, U = pca(reps, 2)
    best_opnorm, best_selected, best_k = -Inf, nothing, nothing
    for k in round.(Int, range(1, 2, length=2))
        selected = rcov_quantum_filter(reps, eps, k, α, τ; limit1=limit1, limit2=limit2)
        Σ = cov(reps_pca[:, selected]')
        Σ′ = cov((Σ^(-1/2)*reps_pca)')
        M = exp(α*(Σ′- I)/(opnorm(Σ′) - 1))
        M /= tr(M)
        op = tr(Σ′ * M) / tr(M)
        poison_removed = sum((.! selected)[end-eps+1:end])
        @show k, op, poison_removed
        if op > best_opnorm
            best_opnorm, best_selected, best_k = op, selected, k
        end

        r = eps*1.5
        poi_list = .! selected
        acc_poi = 0
        miss_poi = 0
        for i in 1:length(poi_indicator)
            if poi_indicator[i]==1
                if poi_list[i]
                    acc_poi += 1
                else
                    miss_poi += 1
                end
            end
        end
        ratio = acc_poi / r
        @printf(log_file, "quantum---PCA-K: %d, opnorm: %d, acc_poi: %d, miss_poi: %d, ratio: %f\n", k, op, acc_poi, miss_poi, ratio)

    end
    @show best_k, best_opnorm
    @printf(log_file, "quantum---best_k: %d, best_opnorm: %d\n", best_k, best_opnorm)
    return best_selected
end


function all_kpca_rcov_quantum_filter(reps, eps, k, reps_pca, α=4, τ=0.1; limit1=2, limit2=1.5)
    d, n = size(reps)
#     reps_pca, U = pca(reps, k)
    if k == 1
        reps_estimated_white = reps_pca
        Σ′ = ones(1, 1)
    else
        selected = cov_estimation_iterate(reps_pca, eps/n, τ, nothing, limit=round(Int, limit1*eps))
        reps_pca_selected = reps_pca[:, selected]
        Σ = cov(reps_pca_selected', corrected=false)
#         reps_estimated_white = Σ^(-1/2)*reps_pca
        reps_estimated_white = sqrt(Hermitian(Σ))\reps_pca
        Σ′ = cov(reps_estimated_white')
    end
    M = k > 1 ? exp(α*(Σ′- I)/(opnorm(Σ′) - 1)) : ones(1, 1)
    M /= tr(M)
    estimated_poison_ind = k_lowest_ind(
        -[x'M*x for x in eachcol(reps_estimated_white)],
        round(Int, limit2*eps)
    )
    return .! estimated_poison_ind
end

function all_kpca_rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator, path, file_name, α=4, τ=0.1; limit1=2, limit2=1.5)
#     reps_path = "$(path)/$(file_name)"
#     reps_pca = npzread(reps_path)
    reps_pca = reps
    best_opnorm, best_selected, best_k = -Inf, nothing, nothing
    for k in round.(Int, range(1, sqrt(100), length=10) .^ 2)

        k_file_name = replace(file_name, "100" => "$(k)")
#         @show "$(path)/$(k_file_name)"
        kpca = npzread("$(path)/$(k_file_name)")
        selected = all_kpca_rcov_quantum_filter(reps, eps, k, kpca, α, τ; limit1=limit1, limit2=limit2)
        Σ = cov(reps_pca[:, selected]')
        Σ′ = cov((Σ^(-1/2)*reps_pca)')
#         Σ′ = cov((sqrt(Hermitian(Σ))\reps_pca)')
        M = exp(α*(Σ′- I)/(opnorm(Σ′) - 1))
        M /= tr(M)
        op = tr(Σ′ * M) / tr(M)
        poison_removed = sum((.! selected)[end-eps+1:end])
        @show k, op, poison_removed
        if op > best_opnorm
            best_opnorm, best_selected, best_k = op, selected, k
        end

        r = eps*1.5
        poi_list = .! selected
        acc_poi = 0
        miss_poi = 0
        for i in 1:length(poi_indicator)
            if poi_indicator[i]==1
                if poi_list[i]
                    acc_poi += 1
                else
                    miss_poi += 1
                end
            end
        end
        ratio = acc_poi / r
        @printf(log_file, "quantum---PCA-K: %d, opnorm: %d, acc_poi: %d, miss_poi: %d, ratio: %f\n", k, op, acc_poi, miss_poi, ratio)
    end
    @show best_k, best_opnorm
    @printf(log_file, "quantum---best_k: %d, best_opnorm: %d\n", best_k, best_opnorm)
    return best_selected
end