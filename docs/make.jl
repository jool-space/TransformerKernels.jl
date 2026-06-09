using TransformerKernels
using Documenter

DocMeta.setdocmeta!(TransformerKernels, :DocTestSetup, :(using TransformerKernels); recursive=true)

makedocs(;
    modules=[TransformerKernels],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="TransformerKernels.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/TransformerKernels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/TransformerKernels.jl",
    devbranch="main",
)
