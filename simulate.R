# try to adapt stochastic block model for multiple types of tie measures
# measure 1: behavior data (gift exchanges)
# measure 2: survey (A says they helped B)

# stochastic block model prototype
library(rethinking)

# simulate sample from model

N_id <- 60L
N_groups <- 3L

# sample ppl into groups
groups <- sample( 1:N_groups , size=N_id , replace=TRUE , prob=c(4,1,1) )

# define interaction matrix across groups
B <- diag(N_groups)
for ( i in 1:length(B) ) if ( B[i]==0 ) B[i] <- 0.05
for ( i in 1:length(B) ) if ( B[i]==1 ) B[i] <- 0.5
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
r_mean <- (-4) # average log-odds report tie (unconditional)
r_1 <- 4 # marginal effect on log-odds when tie is real

# make a tensor for survey responses
# row is focal i and column target j and 3rd index is direction ( i->j or j->i )
s <- array( NA , dim=c(N_id,N_id,2) )

for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        # sim i->j
        s[ i , j , 1 ] <- rbern( 1 , inv_logit( r_mean + r_1*y_true[i,j] ) )
        s[ i , j , 2 ] <- rbern( 1 , inv_logit( r_mean + r_1*y_true[j,i] ) )
    }#j
}#i

# sim gift observations
N_gifts <- 20
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

m <- stan( file="CSBM2.stan" , data=dat , chains=3 , cores=3 , iter=600 )

precis(m,2)

tracerplot(m)

# plot true out network
blank2(w=2)
par(mfrow=c(1,2))

library(igraph)
m_graph <- graph_from_adjacency_matrix( s[,,1] , mode="directed" )
plot(m_graph , vertex.color=groups , main="truth")

# plot posterior inferred network
post <- extract.samples(m)
p_tie_out <- round( apply( post$p_tie_out , 2:3 , mean ) )
m_graph_est <- graph_from_adjacency_matrix( p_tie_out , mode="directed" )
plot(m_graph_est , vertex.color=groups , main="posterior mean" )

# now without known groups

datu <- list(
    N_id = N_id,
    N_groups = N_groups,
    N_gifts = N_gifts,
    group = as.integer( ifelse( runif(length(groups)) < 0 , groups , -1 ) ),
    s = (s),
    g = (g)
)
datu$group[1] <- 1 # fix first individual

mu <- stan( file="CSBM2u.stan" , data=datu , iter=1000 , chains=3 , cores=3 , control=list(adapt_delta=0.95) )

precis(mu,3)

tracerplot(mu)
trankplot(mu)
