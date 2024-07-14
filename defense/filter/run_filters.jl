using Printf
using NPZ
include("util.jl")
include("kmeans_filters.jl")
include("quantum_filters.jl")


for name in ARGS
    path = split(name, "*")[1]
    file_name = split(name, "*")[2]
    eps = parse(Int64, split(name, "*")[3])
    suffix = split(name, "*")[4]


    log_file = open("$(path)/run_filters-$(suffix).log", "w")
    reps_path = "$(path)/$(file_name)"

    poi_indicator_path = "$(path)/$(split(file_name, '-')[1])-poison_indicator.npy"

    reps = npzread(reps_path)
    poi_indicator = npzread(poi_indicator_path)

    n = size(reps)[2]

    eps = round(Int, eps / 1.5)
    removed = round(Int, 1.5*eps)


    @printf("%s: Running PCA filter\n", path)
    reps_pca, U = pca(reps, 1)
#     @printf("%d: length\n", length(reps))
#     @printf("%d: length\n", length(reps_pca))
#     @printf("%d: length\n", length(-abs.(mean(reps_pca[1, :]) .- reps_pca[1, :])))

    pca_poison_ind = k_lowest_ind(-abs.(mean(reps_pca[1, :]) .- reps_pca[1, :]), round(Int, 1.5*eps))
    poison_removed = sum(pca_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    @printf(log_file, "%s-pca: %d, %d\n", path, poison_removed, clean_removed)
    npzwrite("$(path)/mask-pca-target-$(suffix).npy", pca_poison_ind)


    @printf("%s: Running kmeans filter\n", path)
    kmeans_k  = 100
    if occursin("tsne",file_name)
        kmeans_k  = 2
    end
    kmeans_poison_ind = .! kmeans_filter2(reps, eps, kmeans_k)
    poison_removed = sum(kmeans_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    @printf(log_file, "%s-kmeans: %d, %d\n", path, poison_removed, clean_removed)
    npzwrite("$(path)/mask-kmeans-target-$(suffix).npy", kmeans_poison_ind)

    @printf("%s: Running quantum filter\n", path)
    quantum_poison_ind = []
    if occursin("tsne",file_name)
        quantum_poison_ind = .! tsne_rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator)
    else
        if occursin("all_kpca",file_name)
            quantum_poison_ind = .! all_kpca_rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator, path, file_name)
        else
            if occursin("pcaall", suffix)
                quantum_poison_ind = .! pcaall_rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator)
            else
                quantum_poison_ind = .! rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator)
            end
        end
    end
#     try
#         quantum_poison_ind = .! rcov_auto_quantum_filter(reps, eps, log_file, poi_indicator)
#     catch y
#         @printf(log_file, "%s\n", y)
#     end

    poison_removed = sum(quantum_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    @printf(log_file, "%s-quantum: %d, %d\n", path, poison_removed, clean_removed)
    npzwrite("$(path)/mask-rcov-target-$(suffix).npy", quantum_poison_ind)
end
