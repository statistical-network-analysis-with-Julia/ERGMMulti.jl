using Documenter
using ERGMMulti

DocMeta.setdocmeta!(ERGMMulti, :DocTestSetup, :(using ERGMMulti); recursive=true)

makedocs(
    sitename = "ERGMMulti.jl",
    modules = [ERGMMulti],
    authors = "Statistical Network Analysis with Julia",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Statistical-network-analysis-with-Julia.github.io/ERGMMulti.jl",
        edit_link = "main",
    ),
    repo = "https://github.com/Statistical-network-analysis-with-Julia/ERGMMulti.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Multi-Network Structures" => "guide/structures.md",
            "Terms" => "guide/terms.md",
            "Model Estimation" => "guide/estimation.md",
            "Multilevel Networks" => "guide/multilevel.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Terms" => "api/terms.md",
            "Estimation" => "api/estimation.md",
        ],
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(
    repo = "github.com/Statistical-network-analysis-with-Julia/ERGMMulti.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev",
        "dev" => "dev",
    ],
    push_preview = true,
)
