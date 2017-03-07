library(ape)

set.seed(1)

number_of_taxa = c(3,5,10,20)
number_of_replicates = 10

for(taxa in number_of_taxa){
  for(replicate in 1:number_of_replicates){
    tree = rcoal(taxa)
    name = sprintf('trees/tree_taxa%d_%d.new', taxa, replicate)
    write.tree(tree, file=name)
  }
}
