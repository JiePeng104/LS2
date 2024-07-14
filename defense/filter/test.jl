using Printf
using NPZ
# include("kmeans_filters.jl")
# include("quantum_filters.jl")


for name in ARGS
    path = split(name, "*")[1]
    file_name = split(name, "*")[2]
    log_file = open("$(path)/run_filters.log", "w")
    reps_path = "$(path)/$(file_name)"
    println(log_file,reps_path)
    println(log_file, path)
    @printf("%s: Running PCA filter\n", name)
    try
        sqrt(-1)
    catch y
        showerror(log_file, y)
        println(y.msg)
#         @printf(log_file, "%s", y.msg)
    end
    close(log_file)
#     reps = npzread("recourd/$(name)/label_$(target_label)_reps.npy")
end