using Documenter, NeuroPlanner

makedocs(sitename="NeuroPlanner.jl",
        pages = [
            "Home" => "index.md",
            "Heuristic" => "heuristic.md",
            "Usage guide" => "usage_guide.md",
            "Theoretical background" => "theory.md",
            "Losses" => "losses.md",
            "Extractors" => "extractors.md",
            "Model representation" => "model_representation.md"
        ])