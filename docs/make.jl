using Documenter
using ParametricOperators

makedocs(
    sitename = "ParametricOperators.jl",
    format = Documenter.HTML(),
    # modules = [ParametricOperators],
    pages=[
        "Introduction" => "index.md",
        "Quick Start" => "quickstart.md",
        "Distribution" => "distribution.md",
        "Examples" => [
            "3D FFT" => "examples/3D_FFT.md",
            "Distributed 3D FFT" => "examples/3D_DFFT.md",
            "3D Conv" => "examples/3D_Conv.md",
            "Distributed 3D Conv" => "examples/3D_DConv.md",
        ],
        "API" => "api.md",
        "Future Work" => "future.md",
        "Citation" => "citation.md"
    ]
)

# Automatically deploy documentation to gh-pages.
deploydocs(
    repo = "github.com/slimgroup/ParametricOperators.jl.git",
    push_preview=true,
    devurl = "dev",
    devbranch = "main",
    versions = ["dev" => "dev", "stable" => "v^"],
)
