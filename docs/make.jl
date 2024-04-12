using Documenter
using ParametricOperators

makedocs(
    sitename = "ParametricOperators",
    format = Documenter.HTML(),
    # modules = [ParametricOperators]
)

# Automatically deploy documentation to gh-pages.
deploydocs(
    repo = "github.com/slimgroup/ParametricOperators.jl.git",
    devurl = "dev",
    devbranch = "add-documentation",
)
