# Load environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Build db structure
using ReactiveAgentsDriver
con = connect_to_database()
create_database_structure(con)

create_town("small", con)

disconnect_from_database!(con)