using Distributed

NUM_CPU_THREADS = Threads.nthreads()
@show NUM_CPU_THREADS

# Example for a standalone server simulation
addprocs(NUM_CPU_THREADS)

@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using ReactiveAgentsDriver


# Run Variables
NETWORK_SCM = true
EPIDEMIC_SCM = true
TOWN_ID = 1
NUM_NETWORKS = 1
NUM_BEHAVIORS = 1
NUM_EPIDEMICS = 100
NETWORK_INITIAL_LENGTH = 30
MASK_PARTITIONS = 1
VAX_PARTITIONS = 1
BEHAVIOR_DISTR_TYPES = [
   ("Random", "Random"),
   # ("Random", "Watts"),
   # ("Watts", "Random"),
   # ("Watts", "Watts")
]


con = connect_to_database()

for behavior_distr in BEHAVIOR_DISTR_TYPES
   run_simulations(
      TOWN_ID,
      ReactiveAgentsDriver.network_parameters(NETWORK_INITIAL_LENGTH),
      NUM_NETWORKS,
      ReactiveAgentsDriver.behavior_parameters(behavior_distr[1], behavior_distr[2], MASK_PARTITIONS, VAX_PARTITIONS),
      NUM_BEHAVIORS,
      NUM_EPIDEMICS,
      con,
      STORE_NETWORK_SCM=NETWORK_SCM,
      STORE_EPIDEMIC_SCM=EPIDEMIC_SCM,
      DISEASE_PARAMS=[DiseaseParameters(infectious_period=10, γ_parameters=[0.8,4.0])],
      BEHAVIORS_PER_BATCH=4
   )
end

disconnect_from_database!(con)

