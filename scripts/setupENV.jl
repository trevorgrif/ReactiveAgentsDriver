# Only need to all this once on a new machine
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()