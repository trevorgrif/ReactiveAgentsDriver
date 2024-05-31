using Distributed

NUM_PROCESSORS = 5 # Optimally set to the number of available (logical processors) - 1

# Example for a standalone server simulation
addprocs(NUM_PROCESSORS*2)

@everywhere using Pkg
@everywhere Pkg.activate(joinpath(@__DIR__,".."))
@everywhere using ReactiveAgentsDriver

# Run Variables
NETWORK_SCM = true
EPIDEMIC_SCM = true
TOWN_ID = 1
NUM_NETWORKS = 5
NUM_BEHAVIORS = 1
NUM_EPIDEMICS = 50
NETWORK_INITIAL_LENGTH = 10
MASK_PARTITIONS = 5
VAX_PARTITIONS = 5
BEHAVIOR_DISTR_TYPES = [
   ("Random", "Random"),
   ("Random", "Watts"),
   ("Watts", "Random"),
   ("Watts", "Watts")
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
      STORE_EPIDEMIC_SCM=EPIDEMIC_SCM
   )
end

disconnect_from_database!(con)

