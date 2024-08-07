""" 
    Run_RuralABM()

Run the RuralABM package with default parameters.

# Arguments
- `SOCIAL_NETWORKS=10`: Multiplicity of creating town social networks (Range: 1 -> infty).
- `NETWORK_LENGTH=30`: Length in days the model will be run to generate a social network (Range: 1 -> infty).
- `MASKING_LEVELS=5`: Evenly split going from 0 to 100 (exclusive) i.e "2" generates [0,50] (Range: 0 -> 100).
- `VACCINATION_LEVELS=5`: Evenly split going from 0 to 100 (exclusive) i.e "4" generates [0,25,50,75] (Range: 0 -> 100).
- `MODEL_RUNS=100`: Multiplicity model runs with disease spread (Range: 1 -> infty).
- `TOWN_NAMES=["Dubois"]`: Towns which will be run. Ensure input data exist for target towns.
- `OUTPUT_TOWN_INDEX=1`: Index value appended to the town name in the output file directory.
- `OUTPUT_DIR="../output": Default output directory location.
"""
function run_ruralABM(;
    SOCIAL_NETWORKS=10,
    NETWORK_LENGTH=30,
    MASKING_LEVELS=5,
    VACCINATION_LEVELS=5,
    DISTRIBUTION_TYPE=[0, 0],
    MODEL_RUNS=100,
    TOWN_NAMES="small",
    STORE_NETWORK_SCM=true,
    STORE_EPIDEMIC_SCM=true,
    NUMBER_WORKERS=5
)

    _run_ruralABM(
        SOCIAL_NETWORKS=SOCIAL_NETWORKS,
        NETWORK_LENGTH=NETWORK_LENGTH,
        MASKING_LEVELS=MASKING_LEVELS,
        VACCINATION_LEVELS=VACCINATION_LEVELS,
        DISTRIBUTION_TYPE=DISTRIBUTION_TYPE,
        MODEL_RUNS=MODEL_RUNS,
        TOWN_NAMES=TOWN_NAMES,
        STORE_NETWORK_SCM=STORE_NETWORK_SCM,
        STORE_EPIDEMIC_SCM=STORE_EPIDEMIC_SCM,
        NUMBER_WORKERS=NUMBER_WORKERS
    )
end

"""
    Run_Query(query; connection = create_default_connection())

Run a query on the database.
"""
function run_query(query::String, db)
    _run_query(query, db)
end


# Make a global variable to store connection details, this will be called 
function create_database_structure(connection)
    _create_database_structure(connection)
end

macro query(query_str, connection_var)
    quote
        run_query($(esc(query_str)), $(esc(connection_var)))
    end
end


# TODO: Update to take in connection variable
function drop_database_structure()
    _drop_database_structure()
end

function analyze_landing(connection=_create_default_connection())
    _load_staging_tables(connection)
end

function analyze_staging(connection=_create_default_connection())
    _load_fact_tables(connection)
end

function export_database(filepath, connection=_create_default_connection())
    _export_database(filepath, connection)
end

function load_exported_db(filepath)
    connection = _create_default_connection()

    # Check if machine is windows or linux
    if Sys.iswindows()
        # Check `data` directory exists
        if !isdir("data")
            mkdir("data")
        end

        # check filepath\schema.sql exists
        if !isfile("$(filepath)\\schema.sql")
            error("$(filepath)\\schema.sql does not exist")
        end

        # read filepath\schema.sql line by line ignoring empty lines
        open("$(filepath)\\schema.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    _run_query("""$line""", connection)
                end
            end
        end

        # check filepath\load.sql exists
        if !isfile("$(filepath)\\load.sql")
            error("$(filepath)\\load.sql does not exist")
        end

        # read filepath\load.sql line by line ignoring empty lines
        open("$(filepath)\\load.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    _run_query("""$line""", connection)
                end
            end
        end
    else
        # Check `data` directory exists
        if !isdir("data")
            mkdir("data")
        end

        # check filepath/schema.sql exists
        if !isfile("$(filepath)/schema.sql")
            error("$(filepath)/schema.sql does not exist")
        end

        # read filepath/schema.sql line by line ignoring empty lines
        open("$(filepath)/schema.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    _run_query("""$line""", connection)
                end
            end
        end

        # check filepath/load.sql exists
        if !isfile("$(filepath)/load.sql")
            error("$(filepath)/load.sql does not exist")
        end

        # read filepath/load.sql line by line ignoring empty lines
        open("$(filepath)/load.sql") do file
            for line in eachline(file)
                if line != "" && line != ";"
                    # run query
                    _run_query("""$line""", connection)
                end
            end
        end
    end
    DBInterface.close(connection)
end

function connect_to_database(filepath=joinpath("data", "ReactiveAgents.duckdb"))
    _create_default_connection(filepath)
end

function disconnect_from_database!(connection)
    DBInterface.close(connection)
end

function export_table(table_name, filepath, connection=_create_default_connection())
    _export_table(table_name, filepath, connection)
end

struct town_parameters
    type::String
end

struct network_parameters
    duration::Int
end

struct behavior_parameters
    maskDistributionType::String
    vaxDistributionType::String
    maskLevels::Int
    vaxLevels::Int
end


"""
    begin_simulations(town_networks:, mask_levels, vaccine_levels, runs, duration_days_network, towns)

Run RuralABM simulations based on the values passed. See documentation of Run_RuralABM for details.
"""
function run_simulations(
    townData::town_parameters,
    networkData::network_parameters,
    networkCount::Int,
    behaviorData::behavior_parameters,
    behaviorCount::Int,
    epidemicCount::Int,
    db::DuckDB.DB; STORE_NETWORK_SCM::Bool=true,
    STORE_EPIDEMIC_SCM::Bool=true
)
    # TODO: Validate database structure

    # Create town
    println("Creating Town")
    townId = create_town(townData.type, db)

    println("Filling Networks")
    networkIds = fill_network_target(townId, networkData.duration, networkCount, db; STORE_NETWORK_SCM=STORE_NETWORK_SCM)

    println("Filling Behaviors")
    behaviorIds = []
    for networkId in networkIds
        networkBehaviorIds = fill_behaved_network_range(networkId, behaviorData.maskDistributionType, behaviorData.vaxDistributionType, behaviorData.maskLevels, behaviorData.vaxLevels, behaviorCount, db)
        push!(behaviorIds, networkBehaviorIds...)
    end

    println("Filling Epidemics")
    epidemicIds = []
    for behaviorId in behaviorIds
        behaviorEpidemicIds = fill_epidemic_target(behaviorId, epidemicCount, db::DuckDB.DB; STORE_EPIDEMIC_SCM=STORE_EPIDEMIC_SCM)
        push!(epidemicIds, behaviorEpidemicIds...)
    end

    db = vacuum_database(db)
    return db
end

function vacuum_database(connection)
    _vacuum_database(connection)
end

function run_simulations(
    townId::Int,
    networkData::network_parameters,
    networkCount::Int,
    behaviorData::behavior_parameters,
    behaviorCount::Int,
    epidemicCount::Int,
    db::DuckDB.DB; STORE_NETWORK_SCM::Bool=true,
    STORE_EPIDEMIC_SCM::Bool=true,
    DISEASE_PARAMS::Vector{DiseaseParameters}=[DiseaseParameters()],
    BEHAVIORS_PER_BATCH::Int=1
)
    # TODO: Validate database structure

    # Set database to use all threads
    run_query("SET threads TO $(Threads.nthreads())", db)

    println("Filling Networks")
    networkIds = fill_network_target(townId, networkData.duration, networkCount, db; STORE_NETWORK_SCM=STORE_NETWORK_SCM)

    println("Filling Behaviors")
    behaviorIds = []
    for networkId in networkIds
        networkBehaviorIds = fill_behaved_network_range(networkId, behaviorData.maskDistributionType, behaviorData.vaxDistributionType, behaviorData.maskLevels, behaviorData.vaxLevels, behaviorCount, db)
        push!(behaviorIds, networkBehaviorIds...)
    end

    println("Filling Epidemics")
    baseModels = []
    for behaviorId in behaviorIds
        model = _get_model_by_behavior_id(behaviorId, db)
        for i in collect(1:length(DISEASE_PARAMS))
            model_cp = deepcopy(model)
            model_cp.disease_parameters = DISEASE_PARAMS[i]

             # Append disease parameters
            query = """
                SELECT DiseaseParameterID FROM DiseaseParameters
                WHERE BetaStart = $(model_cp.disease_parameters.βrange[1])
                AND BetaEnd = $(model_cp.disease_parameters.βrange[2])
                AND ReinfectionProbability = $(model_cp.disease_parameters.rp)
                AND VaxInfectionProbability = $(model_cp.disease_parameters.vip)
                AND InfectiousPeriod = $(model_cp.disease_parameters.infectious_period)
                AND Gamma1 = $(model_cp.disease_parameters.γ_parameters[1])
                AND Gamma2 = $(model_cp.disease_parameters.γ_parameters[2])
                AND RateOfDecay = $(model_cp.disease_parameters.rate_of_decay)
            """
            result = run_query(query, db)

            diseaseParameterID = 0
            if size(result)[1] !== 0
                diseaseParameterID = result[1,1]
            else 
                query = "SELECT nextval('DiseaseParametersSequence')"
                diseaseParameterID = run_query(query, db)[1,1]
                Rnot = R0(model_cp.disease_parameters.infectious_period, model_cp.disease_parameters.γ_parameters, model_cp.disease_parameters.rate_of_decay)

                appender = DuckDB.Appender(db, "DiseaseParameters")
                DuckDB.append(appender, diseaseParameterID)
                DuckDB.append(appender, model_cp.disease_parameters.βrange[1])
                DuckDB.append(appender, model_cp.disease_parameters.βrange[2])
                DuckDB.append(appender, model_cp.disease_parameters.infectious_period)
                DuckDB.append(appender, model_cp.disease_parameters.γ_parameters[1])
                DuckDB.append(appender, model_cp.disease_parameters.γ_parameters[2])
                DuckDB.append(appender, model_cp.disease_parameters.rp)
                DuckDB.append(appender, model_cp.disease_parameters.vip)
                DuckDB.append(appender, model_cp.disease_parameters.rate_of_decay)
                DuckDB.append(appender, Rnot)
                DuckDB.end_row(appender)
                DuckDB.close(appender)
            end
            model_cp.disease_id = diseaseParameterID

            push!(baseModels, model_cp)
        end
    end

    total_time_start = now()
    for modelBatch in collect(Iterators.partition(baseModels, BEHAVIORS_PER_BATCH))
        # Compute the number of epidemics being ran
        numBehaviors = length(modelBatch)
        numEpidemics = epidemicCount * numBehaviors
        batchStartTime = now()

        # Run numEpidemics for each behavior over the cluster --> flatten results
        # Since we don't store the models after the epidemic, we shouldn't return the entire model from 'fill_epidemic_target'
        epidemicRunStartTime = now()

        # Run epidemics
        # models = fill_epidemic_target(modelBatch[1], epidemicCount, STORE_EPIDEMIC_SCM)
        # models = pmap(fill_epidemic_target, modelBatch, [epidemicCount for _ in 1:numBehaviors], [STORE_EPIDEMIC_SCM for _ in 1:numBehaviors];)
        # models = reduce(vcat, models)
        
        # New Methods:
        # Run all epidemics in modelBatch with pmap
        process = (model) -> begin
            modelCopy = deepcopy(model)
            infect!(modelCopy, 1)
            simulate!(modelCopy)
        end
        models = pmap(process, [modelBatch[(i%numBehaviors) + 1] for i in 0:(epidemicCount*numBehaviors-1)])
        epidemicProcessTime = now()-epidemicRunStartTime


        epidemicIDStart = now()
        for model in models
            id = run_query("SELECT nextval('EpidemicDimSequence')", db)[1, 1]
            model.epidemic_id = id
        end
        epidemicIDProcessTime = now()-epidemicIDStart

        epidemicWriteStart = now()
        # Multi-threaded write to db
        fetch.([Threads.@spawn _append_epidemic_level_data(model, STORE_EPIDEMIC_SCM, db) for model in models]);

        # Single threaded write to db
        # for model in models
        #     _append_epidemic_level_data(model, STORE_EPIDEMIC_SCM, db)
        # end

        # Multi-process write to db
        # epidemicIds = pmap(_append_epidemic_level_data, models, [STORE_EPIDEMIC_SCM for _ in 1:length(models)], [db for _ in 1:length(models)]; distributed=false)
        
        epidemicWriteProcessTime = now()-epidemicWriteStart

        batchDuration = now() - batchStartTime
        println("Finished Batch ($(numEpidemics) epidemics): \n\tTotal Time: $(batchDuration.value/1000) secs \n\tEpidemics/sec: $(numEpidemics/(batchDuration.value/1000)) \n\tEpidemic Process Time: $(epidemicProcessTime.value/1000) secs \n\tID Gather Time: $(epidemicIDProcessTime.value/1000) secs \n\tEpidemic Write Time: $(epidemicWriteProcessTime.value/1000) secs")
    end
    println("Total Time: $(now()-total_time_start)")

    # connection = vacuum_database(connection)
    return db
end

function create_town(town_type::String, connection::DuckDB.DB)
    _create_town!(town_type, connection)
end

function fill_network_target(townId::Int, duration::Int, targetNetworkAmount::Int, connection::DuckDB.DB; STORE_NETWORK_SCM=false)
    # Data Validation
    @assert (targetNetworkAmount >= 0) "Target network amount must be positive: $targetNetworkAmount"
    @assert (duration > 0) "Network duration length must be greater than 0: $targetNetworkAmount"

    model = _get_model_by_town_id(townId, connection)
    model === nothing && return []

    # Compute number of networks needed to run
    query = """
        SELECT NetworkID 
        FROM NetworkDim
        WHERE TownID = $townId
        and ConstructionLengthDays = $duration
    """
    networkIds = run_query(query, connection)[!, 1] .|> Int
    numberNetworks = length(networkIds)
    (numberNetworks >= targetNetworkAmount) && return networkIds

    networkRuns = targetNetworkAmount - numberNetworks
    for _ in 1:networkRuns
        @show "Adding a network"
        networkId = _create_network!(deepcopy(model), duration, connection, STORE_NETWORK_SCM=STORE_NETWORK_SCM)
        push!(networkIds, networkId)
    end

    return networkIds
end

function fill_behaved_network_target(networkId::Int, maskDistributionType::String, vaxDistributionType::String, maskPortion::Int, vaxPortion::Int, targetBehavedNetworkAmount::Int, connection::DuckDB.DB)
    # Data Validation
    @assert (targetBehavedNetworkAmount >= 0) "Target network amount must be positive: $targetBehavedNetworkAmount"
    @assert (0 <= maskPortion <= 100) "Mask portion must be between 0 and 100: $maskPortion"
    @assert (0 <= vaxPortion <= 100) "Vax portion must be between 0 and 100: $vaxPortion"

    model = _get_model_by_network_id(networkId, connection)
    model === nothing && return []

    # Compute number of networks needed to run
    query = """
        SELECT BehaviorID 
        FROM BehaviorDim
        WHERE NetworkID = $networkId
        AND MaskPortion = $maskPortion
        AND VaxPortion = $vaxPortion
        AND MaskDistributionType = '$maskDistributionType'
        AND VaxDistributionType = '$vaxDistributionType'
    """
    behaviorIds = run_query(query, connection)[!, 1] .|> Int
    numberBehaviors = length(behaviorIds)
    (numberBehaviors >= targetBehavedNetworkAmount) && return behaviorIds

    behavedNetworkRuns = targetBehavedNetworkAmount - numberBehaviors
    for _ in 1:behavedNetworkRuns
        behaviorId = _create_behaved_network!(deepcopy(model), maskDistributionType, vaxDistributionType, maskPortion, vaxPortion, connection)
        push!(behaviorIds, behaviorId)
    end

    return behaviorIds
end

function fill_behaved_network_target(model, maskDistributionType::String, vaxDistributionType::String, maskPortion::Int, vaxPortion::Int, targetBehavedNetworkAmount::Int, connection::DuckDB.DB)
    # Data Validation
    @assert (targetBehavedNetworkAmount >= 0) "Target network amount must be positive: $targetBehavedNetworkAmount"
    @assert (0 <= maskPortion <= 100) "Mask portion must be between 0 and 100: $maskPortion"
    @assert (0 <= vaxPortion <= 100) "Vax portion must be between 0 and 100: $vaxPortion"

    # Compute number of behaviors needed to run
    query = """
        SELECT BehaviorID 
        FROM BehaviorDim
        WHERE NetworkID = $(model.network_id)
        AND MaskPortion = $maskPortion
        AND VaxPortion = $vaxPortion
        AND MaskDistributionType = '$maskDistributionType'
        AND VaxDistributionType = '$vaxDistributionType'
    """
    behaviorIds = run_query(query, connection)[!, 1] .|> Int
    numberBehaviors = length(behaviorIds)
    (numberBehaviors >= targetBehavedNetworkAmount) && return behaviorIds

    behavedNetworkRuns = targetBehavedNetworkAmount - numberBehaviors
    for _ in 1:behavedNetworkRuns
        behaviorId = _create_behaved_network!(deepcopy(model), maskDistributionType, vaxDistributionType, maskPortion, vaxPortion, connection)
        push!(behaviorIds, behaviorId)
    end

    return behaviorIds
end

function fill_behaved_network_range(networkId::Int, maskDistributionType::String, vaxDistributionType::String, maskLevels::Int, vaxLevels::Int, targetBehavedNetworkAmount::Int, connection::DuckDB.DB)
    @assert (0 < maskLevels <= 100) "Mask levels must be an integer between 1 and 100: $maskLevels"
    @assert (0 < vaxLevels <= 100) "Vaccine levels must be an integer between 1 and 100: $vaxLevels"
    @assert (targetBehavedNetworkAmount >= 0) "Target behaved network amount must be positive: $targetBehavedNetworkAmount"

    model = _get_model_by_network_id(networkId, connection)
    model === nothing && return []

    mask_increment = floor(100 / maskLevels)
    vax_increment = floor(100 / vaxLevels)

    behaviorIds = []
    for mask_step in 0:(maskLevels)
        for vax_step in 0:(vaxLevels)
            behaviorId = fill_behaved_network_target(model, maskDistributionType, vaxDistributionType, Int(mask_step * mask_increment), Int(vax_step * vax_increment), targetBehavedNetworkAmount, connection)
            push!(behaviorIds, behaviorId...)
        end
    end

    return behaviorIds
end

function fill_epidemic_target(model, targetEpidemicAmount::Int, STORE_EPIDEMIC_SCM=false)
    @assert (targetEpidemicAmount >= 0) "Target epidemic amount must be positive: $targetEpidemicAmount"
    models = _create_epidemic_distributed!(model, targetEpidemicAmount, STORE_EPIDEMIC_SCM=STORE_EPIDEMIC_SCM)
    return models
end