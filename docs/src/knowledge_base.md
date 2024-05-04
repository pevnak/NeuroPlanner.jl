## Knowledge Base

The knowledge base is a structure, which (partially) expresses the computational graph. It is build on top of [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) adding a missing feature to point to the array containing the data (`Mill.jl` strictly supports only computational trees). This enables the KnowledgeBase to reuse computation (the computational graph is directed acyclic graph), which is useful for expressing operation over graphs and hyper/multi graphs.

Before going into details, let start with a simple example implementing a simple graph neural network.

Let's assume the graph has `5` vertices, each desribed by `3` features
```julia
using NeuroPlanner
using NeuroPlanner.Mill
using NeuroPlanner.Flux
x = randn(Float32, 3, 5)
```
Now, let's the graph has 4 edges [(1,2), (2,3), (3,4),(4,1)]. The simple graph neural network will first take features of corresponding vertices, concatenates them (`vcat`), and then it will aggregate them to each node. 

Concatenation of information about the edges is done using `ProductNode` from `Mill.jl` and it will concatenate the source and destination of each edge. 
```julia
xv = ProductNode((KBEntry(:x, [1,2,3,4]), KBEntry(:x, [2,3,4,1])))
```
Notice the use of `KBEntry`, which is a way how to refer to the data in the knowledge base (not yet constructed).

Then, each node needs to aggregate information about each edge. This is achieved by `BagNode` (again from `Mill.jl`), which identifies from which columns of `xv` each vertex receives information. In our case, the first vertex receives information from columns `(1,4)`, the second from `(2,3)` etc. Effectively, the vertex receives information from each edge it contributes to. Therefore a simple message pass can be defined as 
```julia 
BagNode(
	ProductNode((
		KBEntry(:x, [1,2,3,4]), 
		KBEntry(:x, [2,3,4,1]))), 
	ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]])
		)
```
Finally, we can define the structured of the sample by wrapping to the KnowledgeBase as follows
```julia
ds = KnowledgeBase((;
	x = x,
	gnn_1 = BagNode(
	ProductNode((
		KBEntry(:x, [1,2,3,4]), 
		KBEntry(:x, [2,3,4,1]))), 
		ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]]))
	))
```
The `KnowledgeBase` defines the structure of the sample. A corresponding `KnowledgeModel` can be defined manually, but we overload `refletinmodel` from `Mill.jl` for a convenience. The model can be defined as 
```julia
model = reflectinmodel(ds, d -> Dense(d, 16, relu);fsm = Dict("" =>  d -> Dense(d, 1)))
```
and we can project the `KnowledgeBase` by `model` by `model(ds),` since model behaves like any other Flux layer.

```julia
julia> model(ds)
1×5 Matrix{Float32}:
 0.0850099  0.0129541  0.127564  0.202375  0.0
```

The construction of graphs can be simplified by `EdgeBuilder.`

While the above construction might seem clunky at first, it is relatively straightforward to create a hyper-graph. Let's again assume five vertices and hyper-edges `[(1,2,3), (2,3,5), (5,4,5)].` This can be expressed by the following KnowledgeBase
```julia
ds = KnowledgeBase((;
	x = x,
	gnn_1 = BagNode(
	ProductNode((
		KBEntry(:x, [1,2,5]), 
		KBEntry(:x, [2,3,4]), 
		KBEntry(:x, [3,5,5]))), 
		ScatteredBags([[1],[1,2],[1,2],[3],[2,3]]))
	))
```
We create a new model and we are ready to go
```julia
model = reflectinmodel(ds, d -> Dense(d, 16, relu);fsm = Dict("" =>  d -> Dense(d, 1)))
model(ds)
```

If you want to add more message passing layers, you just repeat the construction and replace the `:x` to refer to preceeding layer as 
```julia
ds = KnowledgeBase((;
	x = x,
	gnn_1 = BagNode(
		ProductNode((
			KBEntry(:x, [1,2,3,4]), 
			KBEntry(:x, [2,3,4,1]))), 
			ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]])
		),
	gnn_2 = BagNode(
		ProductNode((
			KBEntry(:gnn_1, [1,2,3,4]), 
			KBEntry(:gnn_1, [2,3,4,1]))), 
			ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]])
		)
	))
```
We create a new model and we are ready to go
```julia
model = reflectinmodel(ds, d -> Dense(d, 16, relu);fsm = Dict("" =>  d -> Dense(d, 1)))
model(ds)
```

Since adding message-passing layers over the same graph is overhead, there is `KBEntryRenamer` which facilitates this. The construction of the above can be simplified to 
```julia
using NeuroPlanner: KBEntryRenamer

mplayer = BagNode(
		ProductNode((
			KBEntry(:x, [1,2,3,4]), 
			KBEntry(:x, [2,3,4,1]))), 
			ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]])
		)

ds = KnowledgeBase((;
	x = x,
	gnn_1 = mplayer,
	gnn_2 = KBEntryRenamer(:x1, :gnn_2)(mplayer),
	))
```

Finally, we can create a multigraph mixing edges with hyperedges as
```julia
mplayer = ProductNode((;
			edges = BagNode(
				ProductNode((
					KBEntry(:x, [1,2,3,4]), 
					KBEntry(:x, [2,3,4,1]))), 
					ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]])
				),
			hyperedges = BagNode(
				ProductNode((
					KBEntry(:x, [1,2,3,4]), 
					KBEntry(:x, [2,3,4,1]))), 
					ScatteredBags([[1,4],[1,2],[2,3],[3,4],Int[]])
				)
			))

ds = KnowledgeBase((;
	x = x,
	gnn_1 = mplayer,
	gnn_2 = KBEntryRenamer(:x1, :gnn_2)(mplayer),
	))
```
The appropriate model can be created by `reflectinmodel.`


### Remarks
  * `KnowledgeBase` behaves like samples, therefore `numobs,`  `Mill.catobs`, and `batch` are overloaded for minibatching. Though we do not implement `MLUtils.getobs,` since that would be quite complicated.
  * `deduplicate` remove duplicities from the knowledgebase without intacting the output.
