# Run analysis on landing tables 

function _load_staging_tables(connection::DuckDB.DB)
    _create_staging_schema(connection)
    _create_agent_view_macro(connection)
    _load_epidemic_results_table(connection)
    _load_behavior_level_aggregate_table(connection)

    _load_epidemic_SCM_degree_distribution(connection)
    _load_epidemic_SCM_weight_distribution(connection)

    _load_network_SCM_degree_distribution(connection)
    _load_network_SCM_weight_distribution(connection)
end

function _create_staging_schema(connection::DuckDB.DB)
    query = """
        CREATE SCHEMA IF NOT EXISTS STG;
    """
    _run_query(query, connection)
end

function _load_epidemic_results_table(connection)
    query = """
        DROP TABLE IF EXISTS STG.EpidemicResults;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.EpidemicResults(
            EpidemicID INTEGER,
            BehaviorID INTEGER, 
            NetworkID INTEGER, 
            TownID INTEGER, 
            MaskPortion DECIMAL, 
            VaxPortion DECIMAL, 
            InfectedTotal INTEGER,
            InfectedMax INTEGER, 
            PeakDay INTEGER, 
            RecoveredTotal INTEGER, 
            RecoveredMasked INTEGER, 
            RecoveredVaccinated INTEGER, 
            RecoveredMaskAndVax INTEGER, 
            MaskedAgentTotal INTEGER, 
            VaxedAgentTotal INTEGER, 
            MaskAndVaxAgentTotal INTEGER, 
            InfectedMaskedAgentTotal INTEGER, 
            InfectedVaxedAgentTotal INTEGER, 
            InfectedMaskAndVaxAgentTotal INTEGER, 
            InfectedMaskAndVaxAgentProbability DECIMAL,
            PRIMARY KEY (EpidemicID, BehaviorID, NetworkID, TownID)
            );
    """
    _run_query(query, connection)

    query = """
    INSERT INTO STG.EpidemicResults
    SELECT DISTINCT
        EpidemicDim.EpidemicID,
        BehaviorDim.BehaviorID,
        NetworkDim.NetworkID,
        TownDim.TownID,
        BehaviorDim.MaskPortion,
        BehaviorDim.VaxPortion,
        EpidemicDim.InfectedTotal,
        EpidemicDim.InfectedMax,
        EpidemicDim.PeakDay,
        EpidemicDim.RecoveredTotal,
        EpidemicDim.RecoveredMasked,
        EpidemicDim.RecoveredVaccinated,
        EpidemicDim.RecoveredMaskAndVax,
        ProtectedTotals.MaskedAgentTotal,
        ProtectedTotals.VaxedAgentTotal,
        ProtectedTotals.MaskAndVaxAgentTotal,
        InfectedTotals.InfectedMaskedAgentTotal,
        InfectedTotals.InfectedVaxedAgentTotal,
        InfectedTotals.InfectedMaskAndVaxAgentTotal,
        CASE 
            WHEN ProtectedTotals.MaskAndVaxAgentTotal = 0 
            THEN 0 
            ELSE CAST(InfectedTotals.InfectedMaskAndVaxAgentTotal AS DECIMAL) / CAST(ProtectedTotals.MaskAndVaxAgentTotal AS DECIMAL) 
            END AS InfectedMaskAndVaxAgentProbability
    FROM EpidemicDim
    LEFT JOIN BehaviorDim ON BehaviorDim.BehaviorID = EpidemicDim.BehaviorID
    LEFT JOIN NetworkDim ON NetworkDim.NetworkID = BehaviorDim.NetworkID
    LEFT JOIN TownDim ON TownDim.TownID = NetworkDim.TownID
    LEFT JOIN (
        -- Get the total number of infected agents in each protected class for each epidemic
        SELECT
            EpidemicDim.BehaviorID,
            EpidemicDim.EpidemicID,
            SUM(CASE WHEN TransmissionLoad.AgentID in (
                SELECT AgentLoad.AgentID
                FROM AgentLoad
                WHERE AgentLoad.IsMasking = 1
                AND AgentLoad.BehaviorID = EpidemicDim.BehaviorID
            ) THEN 1 ELSE 0 END) AS InfectedMaskedAgentTotal,
            SUM(CASE WHEN TransmissionLoad.AgentID in (
                SELECT AgentLoad.AgentID
                FROM AgentLoad
                WHERE AgentLoad.IsVaxed = 1
                AND AgentLoad.BehaviorID = EpidemicDim.BehaviorID
            ) THEN 1 ELSE 0 END) AS InfectedVaxedAgentTotal,
            SUM(CASE WHEN TransmissionLoad.AgentID in (
                SELECT AgentLoad.AgentID
                FROM AgentLoad
                WHERE AgentLoad.IsMasking = 1 AND AgentLoad.IsVaxed = 1
                AND AgentLoad.BehaviorID = EpidemicDim.BehaviorID
            ) THEN 1 ELSE 0 END) AS InfectedMaskAndVaxAgentTotal
        FROM EpidemicDim
        LEFT JOIN TransmissionLoad
        ON TransmissionLoad.EpidemicID = EpidemicDim.EpidemicID
        GROUP BY EpidemicDim.BehaviorID, EpidemicDim.EpidemicID
    ) InfectedTotals
    ON InfectedTotals.EpidemicID = EpidemicDim.EpidemicID
    LEFT JOIN (
        -- Get the total number of agents in each protected class
        SELECT DISTINCT
            AgentLoad.BehaviorID,
            SUM(CASE WHEN AgentLoad.IsMasking = 1 THEN 1 ELSE 0 END) AS MaskedAgentTotal,
            SUM(CASE WHEN AgentLoad.IsVaxed = 1 THEN 1 ELSE 0 END) AS VaxedAgentTotal,
            SUM(CASE WHEN AgentLoad.IsVaxed = 1 THEN AgentLoad.IsMasking ELSE 0 END) AS MaskAndVaxAgentTotal
        FROM AgentLoad
        GROUP BY AgentLoad.BehaviorID
    ) ProtectedTotals
    ON ProtectedTotals.BehaviorID = InfectedTotals.BehaviorID
    """
    _run_query(query, connection)

    return true
end

function _create_agent_view_macro(connection)
    # Load the table
    @query """
    CREATE OR REPLACE MACRO STG.agent_view(EpidemicID) AS TABLE
    SELECT 
        AgentLoad.AgentID,
        EpidemicDim.EpidemicID,
        PopulationLoad.HouseID,
        AgentLoad.AgentHouseholdID,
        PopulationLoad.AgeRangeID,
        PopulationLoad.Sex,
        AgentLoad.IsMasking,
        AgentLoad.IsVaxed,
        CASE WHEN TransmissionLoad.AgentID IS NULL THEN 0 ELSE 1 END AS Infected
    FROM AgentLoad
    JOIN EpidemicDim ON EpidemicDim.BehaviorID = AgentLoad.BehaviorID
    JOIN BehaviorDim ON AgentLoad.BehaviorID = BehaviorDim.BehaviorID
    JOIN NetworkDim ON NetworkDim.NetworkID = BehaviorDim.NetworkID
    JOIN TownDim ON TownDim.TownID = NetworkDim.TownID
    JOIN PopulationLoad ON PopulationLoad.PopulationID = TownDim.PopulationID AND PopulationLoad.AgentID = AgentLoad.AgentID
    LEFT JOIN TransmissionLoad ON TransmissionLoad.AgentID = AgentLoad.AgentID AND TransmissionLoad.EpidemicID = EpidemicDim.EpidemicID
    WHERE EpidemicDim.EpidemicID = EpidemicID
    """ connection
end

function _load_behavior_level_aggregate_table(connection)
    # Drop the table if exist
    query = """
        DROP TABLE IF EXISTS STG.BehaviorLevelAggregate;
    """
    _run_query(query, connection)

    # Create the table
    query = """
        CREATE TABLE STG.BehaviorLevelAggregate(
            BehaviorID INTEGER,
            MaskDistributionType TEXT,
            VaxDistributionType TEXT,
            MaskPortion DECIMAL, 
            VaxPortion DECIMAL,
            IsMaskingCount INTEGER,
            IsVaxedCount INTEGER,
            IsMaskingVaxedCount INTEGER,
            OutbreakSuppresionCount INTEGER,
            OutbreakCount INTEGER,
            ProbabilityOfOutbreak DECIMAL,
            AverageMaskedVaxedInfectedCount DECIMAL,
            ProbabilityOfInfectionWhenProtected DECIMAL,
            AverageInfectedTotal DECIMAL, 
            AverageInfectedPercentage DECIMAL,
            VarianceInfectedTotal DECIMAL,
            AverageInfectedMax DECIMAL, 
            VarianceInfectedMax DECIMAL,
            AveragePeakDay DECIMAL, 
            VariancePeakDay DECIMAL,
            RatioInfectionDeaths DECIMAL
            );
    """
    _run_query(query, connection)

    query = """
    WITH InfectedAndProtectedAgents AS (
        SELECT 
            EpidemicDim.EpidemicID,
            EpidemicDim.BehaviorID,
            SUM(AgentView.Infected) AS ProtectedAndInfectedCount 
        FROM EpidemicDim 
        JOIN STG.agent_view(EpidemicDim.EpidemicID) AgentView ON AgentView.EpidemicID = EpidemicDim.EpidemicID 
        WHERE AgentView.IsMasking = 1 AND AgentView.IsVaxed = 1 
        GROUP BY EpidemicDim.EpidemicID, EpidemicDim.BehaviorID
    ),
    OutbreakSupressionCounts AS (
        SELECT 
           EpidemicDim.BehaviorID,
           (AdultCount + ElderCount + ChildCount) AS Population,
           SUM(CASE WHEN InfectedTotal <= ((AdultCount + ElderCount + ChildCount) * 0.1) THEN 1 ELSE 0 END) AS OutbreakSuppresionCount,
           SUM(CASE WHEN InfectedTotal > ((AdultCount + ElderCount + ChildCount) * 0.1) THEN 1 ELSE 0 END) AS OutbreakCount
        FROM EpidemicDim
        JOIN BehaviorDim ON BehaviorDim.BehaviorID = EpidemicDim.BehaviorID
        JOIN NetworkDim ON NetworkDim.NetworkID = BehaviorDim.NetworkID
        JOIN TownDim ON TownDim.TownID = NetworkDim.TownID
        GROUP BY EpidemicDim.BehaviorID, AdultCount, ElderCount, ChildCount
    ),
    MaskVaxCounts AS (
        SELECT 
            BehaviorID,
            SUM(IsMasking) AS IsMaskingCount,
            SUM(IsVaxed) AS IsVaxedCount,
            SUM(CASE WHEN IsVaxed = 1 THEN IsMasking ELSE 0 END) AS IsMaskingAndVaxed
        FROM AgentLoad
        GROUP BY BehaviorID
    ),
    AggregateInfectedAndProtectedCount AS (
        SELECT 
            InfectedAndProtectedAgents.BehaviorID,
            AVG(ProtectedAndInfectedCount) AS AverageMaskedVaxedInfectedCount
        FROM InfectedAndProtectedAgents
        GROUP BY InfectedAndProtectedAgents.BehaviorID
    ),
    EpidemicData AS (
        SELECT 
            EpidemicDim.BehaviorID,
            AVG(InfectedTotal) AS AverageInfectedTotal, 
            AVG(InfectedTotal)/Population AS AverageInfectedPercentage,
            var_samp(InfectedTotal) AS VarianceInfectedTotal,
            AVG(InfectedMax) AS AverageInfectedMax, 
            var_samp(InfectedMax) AS VarianceInfectedMax,
            AVG(PeakDay) AS AveragePeakDay, 
            var_samp(PeakDay)  As VariancePeakDay,
            AVG(CAST(InfectedTotal AS DECIMAL) / (Population - InfectedTotal + RecoveredTotal)) AS RatioInfectionDeaths
        FROM EpidemicDim
        JOIN OutbreakSupressionCounts ON OutbreakSupressionCounts.BehaviorID = EpidemicDim.BehaviorID
        GROUP BY EpidemicDim.BehaviorID, Population
    )
    INSERT INTO STG.BehaviorLevelAggregate
    SELECT 
        BehaviorDim.BehaviorID,
        BehaviorDim.MaskDistributionType,
        BehaviorDim.VaxDistributionType,
        BehaviorDim.MaskPortion, 
        BehaviorDim.VaxPortion,
        IsMaskingCount,
        IsVaxedCount,
        IsMaskingAndVaxed AS IsMaskingVaxedCount,
        OutbreakSuppresionCount,
        OutbreakCount,
        CAST(OutbreakCount AS DECIMAL)/(OutbreakCount + OutbreakSuppresionCount) AS ProbabilityOfOutbreak,
        AverageMaskedVaxedInfectedCount,
        AverageMaskedVaxedInfectedCount/IsMaskingVaxedCount AS ProbabilityOfInfectionWhenProtected,
        AverageInfectedTotal, 
        AverageInfectedPercentage,
        VarianceInfectedTotal,
        AverageInfectedMax, 
        VarianceInfectedMax,
        AveragePeakDay, 
        VariancePeakDay,
        RatioInfectionDeaths
    FROM BehaviorDim
    LEFT JOIN MaskVaxCounts ON MaskVaxCounts.BehaviorID = BehaviorDim.BehaviorID
    LEFT JOIN OutbreakSupressionCounts ON OutbreakSupressionCounts.BehaviorID = BehaviorDim.BehaviorID
    LEFT JOIN AggregateInfectedAndProtectedCount ON AggregateInfectedAndProtectedCount.BehaviorID = BehaviorDim.BehaviorID
    LEFT JOIN EpidemicData ON EpidemicData.BehaviorID = BehaviorDim.BehaviorID
    ORDER BY 1
    """
    _run_query(query, connection)
end

function _load_network_SCM_weight_distribution(connection)
    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.NetworkSCMWeightDistribution;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.NetworkSCMWeightDistribution(
            NetworkID INTEGER,
            Weight INTEGER,
            WeightCount INTEGER
            );
    """
    _run_query(query, connection)

    # Load the table
    query = """
        INSERT INTO STG.NetworkSCMWeightDistribution
        SELECT 
            NetworkID,
            Weight,
            COUNT(Weight) AS WeightCount
        FROM NetworkSCMLoad
        GROUP BY NetworkID, Weight
    """
    _run_query(query, connection)
end

function _load_epidemic_SCM_weight_distribution(connection)
    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.EpidemicSCMWeightDistribution;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.EpidemicSCMWeightDistribution(
            EpidemicID INTEGER,
            Weight INTEGER,
            WeightCount INTEGER
            );
    """
    _run_query(query, connection)

    # Load the table
    query = """
        INSERT INTO STG.EpidemicSCMWeightDistribution
        SELECT 
            EpidemicID,
            Weight,
            COUNT(Weight) AS WeightCount
        FROM EpidemicSCMLoad
        GROUP BY EpidemicID, Weight
    """
    _run_query(query, connection)
end

function _load_epidemic_SCM_degree_distribution(connection)
    BATCH_SIZE = 100

    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.EpidemicSCMDegreeDistribution;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.EpidemicSCMDegreeDistribution(
            EpidemicID INTEGER,
            Degree INTEGER,
            DegreeCount INTEGER
            );
    """
    _run_query(query, connection)

    # Get a list of EpidemicIDs
    query = """
        SELECT DISTINCT EpidemicID FROM EpidemicDim
    """
    EpidemicIDs = _run_query(query, connection)

    # Break the epidemicIDs into 10 batches and batch load the degree distribution
    for i in 1:BATCH_SIZE:size(EpidemicIDs)[1]
        batch = EpidemicIDs[i:min(i + (BATCH_SIZE - 1), size(EpidemicIDs)[1]), 1]
        query = """
        WITH FullAgentList AS (
            SELECT 
                * 
            FROM EpidemicSCMLoad 
            WHERE Weight <> 0 
            UNION 
            SELECT 
                EpidemicID, 
                Agent2, 
                Agent1, 
                Weight 
            FROM EpidemicSCMLoad
            WHERE Weight <> 0 
        ), AgentDegree AS (
            SELECT 
                EpidemicID, 
                Agent1, 
                COUNT(Agent1) AS Degree 
            FROM FullAgentList 
            WHERE EpidemicID IN (""" * join(Int.(batch), ",") * """)
             GROUP BY EpidemicID, Agent1
         )
         INSERT INTO STG.EpidemicSCMDegreeDistribution
         SELECT EpidemicID, Degree, COUNT(Degree) AS DegreeCount 
         FROM AgentDegree
         GROUP BY EpidemicID, Degree
         """

        _run_query(query, connection)
    end
end

function _load_network_SCM_degree_distribution(connection)
    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.NetworkSCMDegreeDistribution;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.NetworkSCMDegreeDistribution(
            NetworkID INTEGER,
            Degree INTEGER,
            DegreeCount INTEGER
            );
    """
    _run_query(query, connection)

    # Load the table
    query = """
    WITH FullAgentList AS (
        SELECT 
            * 
        FROM NetworkSCMLoad 
        WHERE Weight <> 0 
        UNION 
        SELECT 
            NetworkID, 
            Agent2, 
            Agent1, 
            Weight 
        FROM NetworkSCMLoad
        WHERE Weight <> 0 
    ), AgentDegree AS (
        SELECT 
            NetworkID, 
            Agent1, 
            COUNT(Agent1) AS Degree 
        FROM FullAgentList 
        GROUP BY NetworkID, Agent1
    )
    INSERT INTO STG.NetworkSCMDegreeDistribution
    SELECT NetworkID, Degree, COUNT(Degree) AS DegreeCount 
    FROM AgentDegree
    GROUP BY NetworkID, Degree
    """
    _run_query(query, connection)
end

function _load_epidemic_small_world(connection)
    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.EpidemicSmallWorld;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.EpidemicSmallWorld(
            EpidemicID INTEGER,
            GlobalClusteringCoefficient FLOAT, 
            RandomGlobalClusteringCoefficient  FLOAT,
            AverageShortestPath FLOAT,
            RandomAverageShortestPath  FLOAT
            );
    """
    _run_query(query, connection)

    # Get a list of EpidemicIDs
    query = """
        SELECT DISTINCT EpidemicID FROM EpidemicDim
    """
    EpidemicIDs = _run_query(query, connection)[!, 1]
    EpidemicIDs = Int.(EpidemicIDs)

    # Load the table
    for EpidemicID in EpidemicIDs
        results = _compute_small_world_statistic(EpidemicID, connection, "Epidemic")
        query = """
        INSERT INTO STG.EpidemicSmallWorld
        VALUES ($EpidemicID, 
        $(results["global_clustering_coefficient"]), 
        $(results["random_global_clustering_coefficient"]), 
        $(results["average_shortest_path"]), 
        $(results["random_average_shortest_path"]))
        """
        _run_query(query, connection)
    end

end

function _load_epidemic_t_test(connection::DuckDB.DB)
    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.EpidemicTTest;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.EpidemicTTest(
            MaskPortion INTEGER,
            VaxPortion INTEGER,
            RRVersus STRING,
            Statistic STRING,
            n_x INTEGER,
            n_y INTEGER,
            xbar FLOAT,
            df FLOAT,
            stderr FLOAT,
            t FLOAT,
            μ0 FLOAT,
            PRIMARY KEY (MaskPortion, VaxPortion, RRVersus, Statistic)
            );
    """
    _run_query(query, connection)

    # Run T-Test
    watts_statistical_test(connection, "InfectedMaskAndVaxAgentProbability")
    watts_statistical_test(connection, "PeakDay")
end

function _load_network_small_world(connection)
    # Create the table if it doesn't exist
    query = """
        DROP TABLE IF EXISTS STG.NetworkSmallWorld;
    """
    _run_query(query, connection)

    query = """
        CREATE TABLE STG.NetworkSmallWorld(
            NetworkID INTEGER,
            GlobalClusteringCoefficient FLOAT, 
            RandomGlobalClusteringCoefficient  FLOAT,
            AverageShortestPath FLOAT,
            RandomAverageShortestPath  FLOAT
            );
    """
    _run_query(query, connection)

    # Get a list of EpidemicIDs
    query = """
        SELECT DISTINCT NetworkID FROM NetworkDim
    """
    NetworkIDs = _run_query(query, connection)[!, 1]
    NetworkIDs = Int.(NetworkIDs)

    # Load the table
    for NetworkID in NetworkIDs
        results = _compute_small_world_statistic(NetworkID, connection, "Network")
        query = """
        INSERT INTO STG.NetworkSmallWorld
        VALUES ($NetworkID, 
        $(results["global_clustering_coefficient"]), 
        $(results["random_global_clustering_coefficient"]), 
        $(results["average_shortest_path"]), 
        $(results["random_average_shortest_path"]))
        """
        _run_query(query, connection)
    end

end

##############


#====================#
# Distance Functions #
#====================#

"""
    geometric_mean(v)

Compute the geometric mean of all vectors
"""
function _geometric_mean(v)
    return sqrt.(prod.(v))
end

"""
    arithmetic_mean(v)

Compute the arithmetic mean of all vectors
"""
function _arithmetic_mean(v)
    return mean.(v)
end

#==================================#
# Accessing Upper Half of a Matrix #
#==================================#

function _get_ArithmeticKeyPoints(n)
    KeyPoints = []
    for i in 1:n
        append!(KeyPoints, _get_S(n, i))
    end
    return KeyPoints
end

function _get_S(n, i)
    return Int((i - 1) * (n + 1 - (i / 2.0)))
end

"""
    get_upper_half(matrix)

Get the upper half of a matrix as a vector.
"""
function _get_upper_half(matrix)
    n = size(matrix)[1]
    vector = []
    for i in 1:n-1
        for j in i+1:n
            push!(vector, matrix[i, j])
        end
    end
    return vector
end

"""
    convert_to_vector(List)

Convert a string of comma separated values to a vector of Int64.
"""
function _convert_to_vector(List)
    return parse.(Int64, split(List, ","))
end

"""
    social_contact_matrix_to_graph(EpidemicID::Int64)

Create a simple weighted graph from a social contact matrix.
"""
function _social_contact_matrix_to_graphs(ID::Int64, con::DuckDB.DB, Level::String)
    if (Level ∉ ["Epidemic", "Network"])
        throw(ArgumentError("Level must be either 'Epidemic' or 'Network'"))
    end

    PopulationSize = 386
    if (Level == "Epidemic")

        # Load the SCM
        query = """
            SELECT * 
            FROM EpidemicSCMLoad
            WHERE EpidemicID = $ID
        """
        SCM = _run_query(query, con)
    elseif(Level == "Network")
        # Load the SCM
        query = """
        SELECT * 
        FROM NetworkSCMLoad
        WHERE NetworkID = $ID
        """
        SCM = _run_query(query, con)
    end

    # Create a simple weighted graph
    g = SimpleWeightedGraph(PopulationSize)
    g_normed = SimpleWeightedGraph(PopulationSize)

    # Loop over the SCM df
    MaxWeight = maximum(SCM[!, :Weight])
    for row in eachrow(SCM)
        # Add an edge between node i and node j with weight matrix[i, j]
        add_edge!(g, row[2], row[3], row[4])

        # Add a normalized edge
        weight = row[4]
        weight = (MaxWeight + 1) - weight
        add_edge!(g_normed, row[2], row[3], weight)
    end

    return g, g_normed
end

"""

Iterate over all triples of the graph G
"""
function _triplets(g::SimpleWeightedGraph)
    # Create a vector to store the triplets
    closed_triplet_weights = []
    triplet_weights = []

    # Loop over all nodes
    for i in 1:nv(g)
        # Get the neighbors of node i
        neighbors = Graphs.neighbors(g, i)
        # Get the number of neighbors
        n = length(neighbors)
        # If the node has less than 2 neighbors, the clustering coefficient is 0
        if n < 2
            continue
        else
            # Get the number of edges between the neighbors of node i
            for j in 1:n
                for k in j+1:n
                    # Triplet detected
                    weight_1 = g.weights[i, neighbors[j]]
                    weight_2 = g.weights[i, neighbors[k]]
                    push!(triplet_weights, [weight_1, weight_2])

                    # Check for triangle
                    if has_edge(g, neighbors[j], neighbors[k])
                        push!(closed_triplet_weights, [weight_1, weight_2])
                    end
                end
            end
        end
    end
    return Dict("triplet_weights" => triplet_weights, "closed_triplet_weights" => closed_triplet_weights)
end


"""
clustering_coef(g::SimpleWeightedGraph; method::Function = geometric_mean)

Compute the global clustering coefficient of a weighted graph. Method can be any function that takes a vector of vectors as input and returns a vector of the same length.

See method::Functino = arithmetic_mean

"""
function _global_clustering_coefficient(g::SimpleWeightedGraph; method::Function=_geometric_mean)
    weights = _triplets(g)
    return sum(method(weights["closed_triplet_weights"])) / sum(method(weights["triplet_weights"]))
end

"""
    compute_small_world_statistic(g::SimpleWeightedGraph)
"""
function _compute_small_world_statistic(ID::Int64, con::DuckDB.DB, Level::String; β::Float64=1.0)
    if (Level ∉ ["Epidemic", "Network"])
        throw(ArgumentError("Level must be either 'Epidemic' or 'Network'"))
    end

    # We work with a normalized version of g where the edge weight = MaxWeight + 1 - weight
    # This way the shortest path is the path with the strongest connections
    g, g_normed = _social_contact_matrix_to_graphs(ID, con::DuckDB.DB, Level)

    # Compute the clustering coefficient and the average shortest path for g
    clustering_coefficient = _global_clustering_coefficient(g)
    average_shortest_path = floyd_warshall_shortest_paths(g_normed).dists |> _get_upper_half |> mean

    # Create a randomized graph with the same number of nodes and edges as g
    average_degree = (sum(degree(g)) / length(degree(g))) |> floor |> Int
    weight_distribution, weights = _get_weight_distribution(g)
    weight_normed_distribution, weights_normed = _get_weight_distribution(g_normed)

    g_random = _watts_strogatz(nv(g), average_degree, β, weight_distribution, weights)
    g_random_normed = _watts_strogatz(nv(g_normed), average_degree, β, weight_normed_distribution, weights_normed)

    # Compute the clustering coefficient and the average shortest path for the randomized graph
    random_clustering_coefficient = _global_clustering_coefficient(g_random)
    random_average_shortest_path = floyd_warshall_shortest_paths(g_random_normed).dists |> _get_upper_half |> mean

    return Dict(
        "global_clustering_coefficient" => clustering_coefficient,
        "random_global_clustering_coefficient" => random_clustering_coefficient,
        "average_shortest_path" => average_shortest_path,
        "random_average_shortest_path" => random_average_shortest_path
    )
end

# Custom Watts Strogatz graph
function _watts_strogatz(
    n::Integer,
    k::Integer,
    β::Real,
    weight_distribution,
    weights;
    is_directed::Bool=false,
    remove_edges::Bool=true,
    rng::Union{Nothing,AbstractRNG}=nothing,
    seed::Union{Nothing,Integer}=nothing,
)
    @assert k < n

    g = SimpleWeightedGraph(n)
    # The ith next vertex, in clockwise order.
    # (Reduce to zero-based indexing, so the modulo works, by subtracting 1
    # before and adding 1 after.)
    @inline target(s, i) = ((s + i - 1) % n) + 1

    # Phase 1: For each step size i, add an edge from each vertex s to the ith
    # next vertex, in clockwise order.

    for i in 1:div(k, 2), s in 1:n
        add_edge!(g, s, target(s, i), weights[findall(!iszero, rand(weight_distribution))[1]])
    end

    # Phase 2: For each step size i and each vertex s, consider the edge to the
    # ith next vertex, in clockwise order. With probability β, delete the edge
    # and rewire it to any (valid) target, chosen uniformly at random.

    rng = Graphs.rng_from_rng_or_seed(rng, seed)
    for i in 1:div(k, 2), s in 1:n

        # We only rewire with a probability β, and we only worry about rewiring
        # if there is some vertex not connected to s; otherwise, the only valid
        # rewiring is to reconnect to the ith next vertex, and there is no work
        # to do.
        (rand(rng) < β && degree(g, s) < n - 1) || continue

        t = target(s, i)

        while true
            d = rand(rng, 1:n)          # Tentative new target
            d == s && continue          # Self-loops prohibited
            d == t && break             # Rewired to original target

            t_w = get_weight(g, s, t)   # Current connection
            d_w = get_weight(g, s, d)   # Potential new connection

            d_w != 0.0 && continue          # Already connected

            if add_edge!(g, s, d, t_w)       # Always returns true for SimpleWeightedGraph
                remove_edges && rem_edge!(g, s, t)     # True rewiring: Delete original edge
                break                                   # We found a valid target
            end
        end
    end
    return g
end

#===========================================#
# Re-creating the Graph from Watts-Strogatz #
#===========================================#

"""
    recreate_smallworld_graph(n = 20, k = 4; weight = 0; sample_size = 10)

Recreates the Watts-Strogatz graph from the original paper with n vertices, k nearest neighbors, and rewiring probability p. 

Set weight to a value other than 0 to create a weighted graph.
"""
function _recreate_smallworld_graph(n=20, k=4; weight_distribution=Multinomial(1, [1.0]), weights=0, sample_size=10)
    plotlyjs()

    if weights == 0
        global_clustering_coefficient_function = Graphs.global_clustering_coefficient
        watts_strogatz_function = Graphs.watts_strogatz
        args = (n, k, 0)
    else
        global_clustering_coefficient_function = _global_clustering_coefficient
        watts_strogatz_function = _watts_strogatz
        args = (n, k, 0, weight_distribution, weights)
    end

    avg_C = zeros(15)
    avg_L = zeros(15)

    # Generate p-values
    p_values = []
    for i in 1:15
        p = i / 15.0
        push!(p_values, p)
    end
    p_values = p_values .- 1
    p_values = p_values .* 4
    p_values = 10 .^ (p_values)
    return p_values

    ring_lattice = watts_strogatz_function(args...)

    C_0 = global_clustering_coefficient_function(ring_lattice)
    L_0 = floyd_warshall_shortest_paths(ring_lattice).dists |> _get_upper_half |> mean

    for i in 1:sample_size
        # Generate a random graph for each p-value
        random_graphs = []
        for p in p_values
            if weights == 0
                args = (n, k, p)
            else
                args = (n, k, p, weight_distribution, weights)
            end
            g = watts_strogatz_function(args...)

            push!(random_graphs, g)
        end

        # Compute L(p) and C(p) for each graph
        L_p = []
        C_p = []
        for g in random_graphs
            L = floyd_warshall_shortest_paths(g).dists |> _get_upper_half |> mean
            C = global_clustering_coefficient_function(g)
            push!(L_p, L)
            push!(C_p, C)
        end

        # Divide each C(p) value by C(0)
        C_p = C_p ./ C_0

        # Divide each L(p) value by L(0)
        L_p = L_p ./ L_0

        avg_C .+= C_p
        avg_L .+= L_p
    end

    avg_C = avg_C ./ sample_size
    avg_L = avg_L ./ sample_size

    # Graph C_p and L_p as two scatter plots 
    if weight == 0
        title = "Non-Weighted Graphs"
    else
        title = "Weighted Graphs"
    end
    x_values = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_display_values = [" ", " ", " ", " ", " ", "0.001", " ", " ", " ", " ", " ", "0.01", " ", " ", " ", " ", " ", "0.1", " ", " ", " ", " ", "1.0"]

    Plots.plot(
        p_values,
        avg_C,
        xscale=:log10,
        label="C(p)/C(0)",
        xlabel="p",
        ylabel="Ratio over Regular Lattice",
        seriestype=:scatter
    )
    Plots.scatter!(
        p_values,
        avg_L,
        xscale=:log10,
        label="L(p)/L(0)",
        xlabel="p",
        ylabel="Ratio over Regular Lattice",
        title=title,
        xticks=(x_values, x_display_values)
    )
end

# Get the weight distribution of a graph G
function _get_weight_distribution(G)
    weights = []
    for e in edges(G)
        push!(weights, e.weight)
    end
    aggWeights = countmap(weights)
    total = sum(values(aggWeights))

    probability_vector = []
    for w in keys(aggWeights)
        push!(probability_vector, aggWeights[w] / total)
    end
    probability_vector = convert.(Float64, probability_vector)

    return Multinomial(1, probability_vector), collect(keys(aggWeights))
end

#==============#
# Watts Effect #
#==============#

"""
    watts_statistical_test(target_distribution = 0)

Perform a statistical t-test (Welch's T-Test) to determine if the average of InfectedMaskAndVaxAgentProbability is significantly different between town builds at the given target distribution.

Returns an array of test results between all four town builds.

UPDATE: Compare between distribution methods not between towns (e.g. RR RW WR WW)
TODO: Each watts variation (WR, RW, WW) to RR, at the epidemic level (aggregate 100 together), for each of the levels (20,40,60)

27 T-Tests total

Also: same thing but with PeakHeight instead of InfectedMaskAndVaxAgentProbability

# Note: target_distribution = 10 * mask_portion + vax_portion
"""
function watts_statistical_test(connection::DuckDB.DB, targetStat::String = "InfectedMaskAndVaxAgentProbability")
    # Get the protected infection probability for epidemic
    query = """
    SELECT 
        EpidemicID,
        EpidemicResults.MaskPortion, 
        EpidemicResults.VaxPortion,
        MaskDistributionType,
        VaxDistributionType,
        $targetStat 
    FROM STG.EpidemicResults
    JOIN BehaviorDim ON BehaviorDim.BehaviorID = EpidemicResults.BehaviorID
    """
    data = _run_query(query, connection)

    # Aggregate by distribution type
    RR = data[(data.MaskDistributionType .== "Random") .& (data.VaxDistributionType .== "Random"), :]
    WR = data[(data.MaskDistributionType .== "Watts") .& (data.VaxDistributionType .== "Random"), :]
    RW = data[(data.MaskDistributionType .== "Random") .& (data.VaxDistributionType .== "Watts"), :]
    WW = data[(data.MaskDistributionType .== "Watts") .& (data.VaxDistributionType .== "Watts"), :]
    
    # Iterate over each mask and vax level and take the t-test between RR and the other three (WR, RW, WW)
    
    v = [20, 40, 60, 80]
    for mask in v, vax in v
        RR_target = RR[RR.MaskPortion .== mask, :]
        WR_target = WR[WR.MaskPortion .== mask, :]
        RW_target = RW[RW.MaskPortion .== mask, :]
        WW_target = WW[WW.MaskPortion .== mask, :]

        RR_target = RR_target[RR_target.VaxPortion .== vax, :]
        WR_target = WR_target[WR_target.VaxPortion .== vax, :]
        RW_target = RW_target[RW_target.VaxPortion .== vax, :]
        WW_target = WW_target[WW_target.VaxPortion .== vax, :]

        result_1 = UnequalVarianceTTest(Float64.(RR_target[!, targetStat]), Float64.(WR_target[!, targetStat]))
        result_2 = UnequalVarianceTTest(Float64.(RR_target[!, targetStat]), Float64.(RW_target[!, targetStat]))
        result_3 = UnequalVarianceTTest(Float64.(RR_target[!, targetStat]), Float64.(WW_target[!, targetStat]))

        # Insert the EpidemicTTest table
        query = """
        INSERT INTO STG.EpidemicTTest
        VALUES 
        ($mask, $vax, 'WR', '$targetStat', $(result_1.n_x), $(result_1.n_y), $(result_1.xbar), $(result_1.df), $(result_1.stderr), $(result_1.t), $(result_1.μ0)),
        ($mask, $vax, 'RW', '$targetStat', $(result_2.n_x), $(result_2.n_y), $(result_2.xbar), $(result_2.df), $(result_2.stderr), $(result_2.t), $(result_2.μ0)),
        ($mask, $vax, 'WW', '$targetStat', $(result_3.n_x), $(result_3.n_y), $(result_3.xbar), $(result_3.df), $(result_3.stderr), $(result_3.t), $(result_3.μ0))
        """
        _run_query(query, connection)
    end

end