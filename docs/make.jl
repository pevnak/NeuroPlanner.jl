using Documenter, NeuroPlanner

using Downloads: download
using Documenter.Writers: HTMLWriter
using DocumenterTools.Themes


pages = [
            "Home" => "index.md",
            "Heuristic" => "heuristic.md",
            "Usage guide" => "usage_guide.md",
            "Theoretical background" => "theory.md",
            "Losses" => "losses.md",
            "Extractors" => "extractors.md",
            "Model representation" => "model_representation.md"
        ]

makedocs(;
    authors = "Tomas Pevny, Daniel Zampach, Viteslav Simek",
    repo = "https://github.com/Pevnak/NeuroPlanner.jl/blob/{commit}{path}#{line}",
    sitename = "NeuroPlanner",
    format = Documenter.HTML(;
        prettyurls = true,
        canonical = "https://Pevnak.github.io/NeuroPlanner",
        assets = ["assets/favicon.ico", "assets/onlinestats.css"],
        collapselevel = 1,
        ansicolor=true,
    ),
    pages
)

deploydocs(
    repo = "github.com/pevnak/NeuroPlanner.jl.git"
)

