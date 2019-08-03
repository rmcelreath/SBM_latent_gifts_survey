# try to adapt stochastic block model for multiple types of tie measures
# measure 1: behavior data (gift exchanges)
# measure 2: survey (A says they helped B)

# stochastic block model prototype
library(rethinking)

# simulate sample from model

N_id <- 60L
N_groups <- 3L

# sample ppl into groups
groups <- sample( 1:N_groups , size=N_id , replace=TRUE , prob=c(3,2,1) )

# define interaction matrix across groups
B <- diag(N_groups)
for ( i in 1:length(B) ) if ( B[i]==0 ) B[i] <- 0.01
for ( i in 1:length(B) ) if ( B[i]==1 ) B[i] <- 0.2
# B[1,2] <- 0.9

# sim ties
y_true <- matrix( 0 , N_id , N_id )
idA <- 0
idB <- 0
for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            y_true[i,j] <- rbern( 1 , B[ groups[i] , groups[j] ] )
        }
    }#j
}#i

# sim survey
# need to define probability of reporting tie, conditional on tie
r_mean <- (-2) # average log-odds report tie (unconditional)
r_1 <- 1 # marginal effect on log-odds when tie is real

# make a tensor for survey responses
# row is focal i and column target j and 3rd index is direction ( i->j or j->i )
s <- array( 0 , dim=c(N_id,N_id,2) )

for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            # sim i->j
            s[ i , j , 1 ] <- rbern( 1 , inv_logit( r_mean + r_1*y_true[i,j] ) )
            s[ i , j , 2 ] <- rbern( 1 , inv_logit( r_mean + r_1*y_true[j,i] ) )
        }
    }#j
}#i

# sim gift observations
N_gifts <- 5
g <- array( 0L , dim=c(N_id,N_id,N_gifts) )
g_mean <- -4
g_1 <- 4

for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i!=j )
            g[ i , j , ] <- rbern( N_gifts , inv_logit( g_mean + g_1*y_true[i,j] ) )
    }#j
}#i

dat <- list(
    N_id = N_id,
    N_groups = N_groups,
    N_gifts = N_gifts,
    group = groups,
    s = (s),
    g = (g)
)

m <- stan( file="CSBM2.stan" , data=dat , chains=3 , cores=3 , iter=1000 )

precis(m,2)
precis(m,3,pars="B")

tracerplot(m)


library(igraph)

m_graph <- graph_from_adjacency_matrix( y_true , mode="directed" )


#Get some network descriptives
#See how long the furthest path is
diameter(m_graph, directed = TRUE, weights = NA)
#Get the nodes involved
get_diameter(m_graph, directed = TRUE, weights = NA)
#Calculate the mean distance between nodes
mean_distance(m_graph, directed = TRUE)
#Get the network density
edge_density(m_graph)
#General tendency towards reciprocity
reciprocity(m_graph)
#General tendency towards transtivity
transitivity(m_graph)
#Range of indegree and outdegree
range(degree(m_graph, mode="in"))
range(degree(m_graph, mode="out"))
# All look pretty plausible

# plot true out network
blank2(w=2)
par(mfrow=c(1,2))

#plot(m_graph, vertex.color=groups , main="posterior mean" )
#Pretty but not sure if super informative 
deg <- 20
plot(m_graph , vertex.color=groups , vertex.size = deg*.4, edge.arrow.size =0.15, 
     edge.curved = 0.35, vertex.label = NA, seed = 1,
     main="truth")
#Decreasing vertex size helps see the direction of ties
#I included the degree (undirected) in the visualisation
#All can be easily modified 

# plot posterior inferred network
post <- extract.samples(m)
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )
m_graph_est <- graph_from_adjacency_matrix( p_tie_out , mode="directed" , weighted=TRUE )

#Get some network descriptives
#See how long the furthest path is
diameter(m_graph_est, directed = TRUE, weights = NA)
#Get the nodes involved
get_diameter(m_graph_est, directed = TRUE, weights = NA)
#Calculate the mean distance between nodes
mean_distance(m_graph_est, directed = TRUE)
#Get the network density
edge_density(m_graph_est)
#General tendency towards reciprocity
reciprocity(m_graph_est)
#General tendency towards transtivity
transitivity(m_graph_est)
#Range of indegree and outdegree
range(degree(m_graph_est, mode="in"))
range(degree(m_graph_est, mode="out"))
#Again, all look pretty plausible

#plot(m_graph_est , vertex.color=groups , main="posterior mean" )

# Again, pretty but maybe not the most informative
plot(m_graph_est ,vertex.color=groups , vertex.size = deg*.3, edge.arrow.size =0.15, 
     edge.curved = 0.35,  vertex.label = NA,  seed = 1, main="posterior mean" )

# plot edge weights
w <- as.vector(pmean)
o <- order(w)
blank2(w=3)
plot( w[o] , xlab="edge (sorted by weight)" , ylab="posterior weight" , col="white" )
pci <- apply( post$p_tie_out , 2:3 , PI , prob=0.95 )
phi <- as.vector(pci[2,,])[o]
plo <- as.vector(pci[1,,])[o]
for ( i in 1:length(phi) ) lines( c(i,i) , c(phi[i],plo[i]) )


# now without known groups

datu <- list(
    N_id = N_id,
    N_groups = N_groups,
    N_gifts = N_gifts,
    group = as.integer( ifelse( runif(length(groups)) < 0.5 , groups , -1 ) ),
    s = (s),
    g = (g)
)
datu$group[1] <- groups[1] # fix first individual

mu <- stan( file="CSBM2u.stan" , data=datu , iter=600 , chains=2 , cores=2 , control=list(adapt_delta=0.95) )

precis(mu,2)
precis(mu,3,pars="B")

tracerplot(mu)
trankplot(mu)

# plot true out network
blank2(w=2)
par(mfrow=c(1,2))

library(igraph)
m_graph <- graph_from_adjacency_matrix( y_true , mode="directed" )
plot(m_graph , vertex.color=groups , main="truth" , edge.arrow.size=0.55 , edge.curved=0.35 , edge.color=gray(0.5) )

# plot posterior inferred network
post <- extract.samples(mu)
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )

gmean <- apply( post$p_group , 2:3 , mean )
gest <- sapply( 1:N_id , function(i) which.max( gmean[i,] ) )
# calculate accuracy
table( gest[datu$group<0]==groups[datu$group<0] )
# cbind( round(gmean,2) , groups , datu$group , groups==gest )

m_graph_est <- graph_from_adjacency_matrix( p_tie_out , mode="directed" , weighted=TRUE )
plot(m_graph_est , vertex.color=gest , vertex.frame.color=groups , main="posterior mean" , edge.arrow.size=0.55 , edge.curved=0.35 , edge.color=gray(0.5) )

# true ties against inferred
post <- extract.samples(mu)
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )
table( y_true , p_tie_out )

