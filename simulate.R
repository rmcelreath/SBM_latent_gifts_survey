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

# varying effects on individuals
# (1) general tendency to form out ties
# (2) general tendency to form in ties
v <- rmvnorm2( N_id , Mu=c(0,0) , sigma=c(1,1) , Rho=diag(2) )

# sim ties
y_true <- matrix( 0 , N_id , N_id )
for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            pij <- inv_logit( logit(B[ groups[i] , groups[j] ]) + v[i,1] + v[j,2] )
            y_true[i,j] <- rbern( 1 , pij )
        }
    }#j
}#i

# set up observable variables
N_x <- 3
alpha <- c( 0 , -2 , -4 )
beta <- c( 2 , 1 , 4 )

# sim outcomes
# survey responses: (1) i->j, (2) j->i (as reported by i)
s <- array( 0L , dim=c( N_id , N_id , 2 ) ) 
# gifts
N_gifts <- 5
g <- array( 0L , dim=c( N_id , N_id , N_gifts ) )

for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            # sim i->j
            s[ i , j , 1 ] <- rbern( 1 , inv_logit( alpha[1] + beta[1]*y_true[i,j] ) )
            s[ i , j , 2 ] <- rbern( 1 , inv_logit( alpha[2] + beta[2]*y_true[j,i] ) )
            g[ i , j , ] <- rbern( N_gifts , inv_logit( alpha[3] + beta[3]*y_true[i,j] ) )
        }
    }#j
}#i

dat <- list(
    N_x = 3,
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

# true ties against inferred
post <- extract.samples(m)
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )
table( y_true , p_tie_out )

# individual effects
v_est <- apply( post$v_id , 2:3 , mean )
blank2(w=2)
par(mfrow=c(1,2))
plot( v[,1] , v_est[,1] )
plot( v[,2] , v_est[,2] )

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

# true ties against inferred
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )
table( y_true , p_tie_out )

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
    N_x = 3,
    N_id = N_id,
    N_groups = 3,
    N_gifts = N_gifts,
    group = as.integer( ifelse( runif(length(groups)) < 0.1 , groups , -1 ) ),
    s = (s),
    g = (g),
    pg_prior = rep(6,3)
)
datu$group[1] <- groups[1] # fix first individual
datu$pg_prior <- rep(10,3)
datu$group

mu <- stan( file="CSBM2u.stan" , data=datu , iter=600 , chains=3 , cores=3 , control=list(adapt_delta=0.95) )

precis(mu,2)
precis(mu,3,pars="B")

tracerplot(mu)
trankplot(mu)


# true ties against inferred
post <- extract.samples(mu)
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )
table( y_true , p_tie_out )


# plot true out network
blank2(w=2,ex=2)

par(mfrow=c(1,2))

library(igraph)
m_graph <- graph_from_adjacency_matrix( y_true , mode="directed" )
lx <- layout_nicely(m_graph)
plot(m_graph , vertex.color=groups , vertex.size=8  , main="truth" , edge.arrow.size=0.55 , edge.curved=0.35 , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 , layout=lx )

# plot posterior inferred network
post <- extract.samples(mu)
pmean <- apply( post$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )

gmean <- apply( post$p_group , 2:3 , mean )
gest <- sapply( 1:N_id , function(i) which.max( gmean[i,] ) )
# prob of each group as list, for vertex.shape="pie"
glist <- lapply( 1:nrow(gmean) , function(i) as.integer( 10L*round( gmean[i,] , 1 ) ) )
# calculate accuracy
table( gest[datu$group<0]==groups[datu$group<0] )
# cbind( round(gmean,2) , groups , datu$group , groups==gest )

m_graph_est <- graph_from_adjacency_matrix( p_tie_out , mode="directed" , weighted=TRUE )
plot( m_graph_est , vertex.shape="pie" , vertex.pie=glist , vertex.pie.color=list(c(1,2,3,4)) , vertex.size=8 , main="posterior mean" , edge.arrow.size=0.55 , edge.curved=0.35 , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 , vertex.label=NA , layout=lx )

# elly's style
blank2(w=4)
par(mfrow=c(1,4))
lx <- layout_nicely(m_graph_est)
plot( m_graph_est , vertex.color=gest , vertex.size=8 , main="posterior mean" , edge.arrow.size=0.55 , edge.curved=0.35 , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 , vertex.label=NA , layout=lx )
for ( i in 1:N_groups ) plot( m_graph_est , vertex.color=gray(1-gmean[,i]) , vertex.size=8 , main=concat("clique ",i) , edge.arrow.size=0.55 , edge.curved=0.35 , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 , vertex.label=NA , layout=lx )

