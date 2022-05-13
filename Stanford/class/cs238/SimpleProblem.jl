struct SimpleProblem
    bn::BayesianNetwork
    chance_vars::Vector{Variable}
    decision_vars::Vector{Variable}
    utility_vars::Vector{Variable}
    utilities::Dict{Symbol,Vector{Float64}}
end
function solve(𝒫::SimpleProblem, evidence, M)
    query = [var.name for var in 𝒫.utility_vars]
    U(a) = sum(𝒫.utilities[uname][a[uname]] for uname in query)
    best = (a = nothing, u = -Inf)
    for assignment in assignments(𝒫.decision_vars)
        evidence = merge(evidence, assignment)
        ϕ = infer(M, 𝒫.bn, query, evidence)
        u = sum(p * U(a) for (a, p) in ϕ.table)
        if u > best.u
            best = (a = assignment, u = u)
        end
    end
    return best
end

O1 = Variable(:o1, 2);
O2 = Variable(:o2, 2);
O3 = Variable(:o3, 2)
D = Variable(:d, 2);
U = Variable(:u, 2);
T = Variable(:t, 2);
vars = [O1, O2, O3, D, U, T]
factors = [
]
graph = SimpleDiGraph(6)
add_edge!(graph, 4, 1);
add_edge!(graph, 4, 2);
add_edge!(graph, 4, 3);
add_edge!(graph, 4, 5);
add_edge!(graph, 6, 5);
bn = BayesianNetwork(vars, factors, graph)

𝒫 = SimpleProblem(
    bn, # BayesianNetwork
    [1, 2, 3, 4], # chance_vars
    [6], # decision_vars
    [5], # utility_vars
    [0 -10 -1 -1]
    )
solve(𝒫, #SimpleProblem 
    Dict(:o1 => 1, :o2 => 1), #evidence 
    Matrix([ 0 0 1 1; 0 1 0 1]) #M
    )
