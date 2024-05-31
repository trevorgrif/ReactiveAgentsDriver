using Distributed

NUM_PROCESSORS = 10 # Optimally set to the number of available (logical processors) - 1

# Example for a standalone server simulation
addprocs(NUM_PROCESSORS)

# Example for a Slurm Cluster simulation
# addprocs(SlurmManager(NUM_PROCESSORS))

@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere using ReactiveAgentsDriver
# @everywhere using ClusterManagers

# Run Variables
NETWORK_SCM = false
EPIDEMIC_SCM = false
TOWN_ID = 1
NUM_NETWORKS = 1
NUM_BEHAVIORS = 1
NUM_EPIDEMICS = 50
NETWORK_INITIAL_LENGTH = 30
MASK_PARTITIONS = 1
VAX_PARTITIONS = 1

disease_params::Vector{DiseaseParameters} = []
ip_range = collect(10:15)
gamma1_range = collect(0.5:0.1:2.0)
for ip in ip_range
    for gamma1 in gamma1_range
        push!(disease_params, DiseaseParameters(infectious_period = ip, γ_parameters = [gamma1, 4.0]))
    end
end

con = connect_to_database()

run_simulations(
    TOWN_ID,
    ReactiveAgentsDriver.network_parameters(NETWORK_INITIAL_LENGTH),
    NUM_NETWORKS,
    ReactiveAgentsDriver.behavior_parameters("Random", "Random", MASK_PARTITIONS, VAX_PARTITIONS),
    NUM_BEHAVIORS,
    NUM_EPIDEMICS,
    con,
    STORE_NETWORK_SCM=NETWORK_SCM,
    STORE_EPIDEMIC_SCM=EPIDEMIC_SCM,
    DISEASE_PARAMS=disease_params
)

disconnect_from_database!(con)
