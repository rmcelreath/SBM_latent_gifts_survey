# try to adapt stochastic block model for multiple types of tie measures
# measure 1: behavior data (gift exchanges)
# measure 2: survey (A says they helped B)

# rm(list = ls())
# setwd()

# stochastic block model prototype

# Load packages
library(igraph)
library(rethinking)
library(Rlab)

# Simulate sample from model

simulate_network <- function( N_id = 60L, N_groups = 3L, gprobs = c(3,1,1),
                                observed_groups = TRUE, in_block = 0.5, out_block = 0.1, 
                                indeg_mu = 0, outdeg_mu = 0, 
                                indeg_sigma = 1, outdeg_sigma = 1,
                                N_x = 3, alpha = c(0, -2, -4), beta = c( 2 , 1 , 4 ),
                                theta = 0.5, N_responses = 2, prob_obs_g = 0.8,
                                N_gifts = 5, bad_people = 1, pg_prior = rep(6,3)
                                )
                                 {
# sample ppl into groups
groups <- sample( 1:N_groups , size=N_id , replace=TRUE , prob=gprobs )

# define interaction matrix across groups
B <- diag(N_groups)
for ( i in 1:length(B) ) if ( B[i]==0 ) B[i] <- out_block
for ( i in 1:length(B) ) if ( B[i]==1 ) B[i] <- in_block
# B[1,2] <- 0.9

# If j -> i is unknown/network is single sampled
if ( N_responses == 1 ) {

# varying effects on individuals
# (1) general tendency to form out ties
v <- rmvnorm2( N_id , Mu=c(outdeg_mu) , sigma=c(outdeg_sigma) , Rho= 1 )


y_true <- matrix( 0 , N_id , N_id )
for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            pij <- inv_logit( logit(B[ groups[i] , groups[j] ]) + v[i,1] )
            y_true[i,j] <- rbern( 1 , pij )
            }
        }#j
    }#i

s <- array( 0L , dim=c( N_id , N_id ,  1 ) ) 
g <- array( 0L , dim=c( N_id , N_id , N_gifts ) )

for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            # only sim i->j
            s[ i , j, 1] <- rbern( 1 , inv_logit( alpha[1] + beta[1]*y_true[i,j] ) )
                if ( i==bad_people) 
                    s[ i , j , 1 ] <- rbern( 1 , inv_logit( 3 + beta[1]*y_true[i,j] ) )
            # then sim gifts 
            g[ i , j , ] <- rep( -1 , N_gifts )
            for ( k in 1:N_gifts ) {
                if ( runif(1) < prob_obs_g )
                    g[ i , j , k ] <- rbern( 1 , inv_logit( alpha[2] + beta[2]*y_true[i,j] ) )
                }#k    
            }
        }#j
    }#i
}

# For double-sampled networks/when j -> i is known  
else{

# varying effects on individuals
# (1) general tendency to form out ties
# (2) general tendency to form in ties
v <- rmvnorm2( N_id , Mu=c( outdeg_mu , indeg_mu ) , sigma=c( outdeg_sigma , indeg_sigma ) , Rho=diag(2) )

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

# sim outcomes
# survey responses: (1) i->j, (2) j->i (as reported by i)
s <- array( 0L , dim=c( N_id , N_id , 2 ) ) 
# gifts
g <- array( 0L , dim=c( N_id , N_id , N_gifts ) )

for ( i in 1:N_id ) {
    for ( j in 1:N_id ) {
        if ( i != j ) {
            # sim i->j
            s[ i , j , 1 ] <- rbern( 1 , inv_logit( alpha[1] + beta[1]*y_true[i,j] ) )
                if ( i==bad_people) 
                    s[ i , j , 1 ] <- rbern( 1 , inv_logit( 3 + beta[1]*y_true[i,j] ) )
            # then sim j -> i 
             s[ i , j , 2 ] <- rbern( 1 , inv_logit( (1-s[i,j,1])*alpha[2] + beta[2]*y_true[j,i] + s[i,j,1]*theta ) )
            # then sim gifts 
            g[ i , j , ] <- rep( -1 , N_gifts )
            for ( k in 1:N_gifts ) {
                if ( runif(1) < prob_obs_g )
                    g[ i , j , k ] <- rbern( 1 , inv_logit( alpha[3] + beta[3]*y_true[i,j] ) )
               }#k
            }
        }#j
    }#i
}

if(observed_groups == TRUE) {
output <- list(
    N_x = N_x,
    N_id = N_id,
    N_groups = N_groups,
    N_gifts = N_gifts,
    group = groups,
    s = (s),
    g = (g),
    v = v, 
    y_true = y_true
        )
}
    else{
        output <- list(
        N_x = N_x,
        N_id = N_id,
        N_groups = N_groups,
        N_gifts = N_gifts,
        group = as.integer( ifelse( runif(length(groups)) < 0.1 , groups , -1 ) ),
        s = (s),
        g = (g),
        v = v, 
        y_true = y_true,
        pg_prior = pg_prior
        )
            output$group[1] <- groups[1] # fix first individual
               
}

return(output)
}

dat <- simulate_network()
datu <- simulate_network(observed_groups = FALSE)

# Check degree distribution
hist(degree(graph_from_adjacency_matrix(as.matrix(dat$y_true))))
hist(degree(graph_from_adjacency_matrix(as.matrix(datu$y_true))))


# Run models
m <- stan( file="CSBM2.stan" , data=dat , chains=3 , cores=3 , iter=1000 )
mu <- stan( file="CSBM2u.stan" , data=datu , iter=600 , chains=3 , cores=3 , control=list(adapt_delta=0.95) )

# Check model with observed groups
precis(m,2)
precis(m,3,pars="B")

# Check model with unobserved groups
precis(mu,2)
precis(mu,3,pars="B")


# Plot chains
tracerplot(m)
tracerplot(mu)
trankplot(m)
trankplot(mu)


mpost <- extract.samples(m)
pmean <- apply( mpost$p_tie_out , 2:3 , mean )
p_tie_out <- round( pmean )

mupost <- extract.samples(mu)
u_pmean <- apply( mupost$p_tie_out , 2:3 , mean )
u_p_tie_out <- round( u_pmean )




# plot grid
blank2(w=2)
par(mfrow=c(1,2))
image(dat$y_true)
image(mpost$pmean)

# plot grid
blank2(w=2)
par(mfrow=c(1,2))
image(datu$y_true)
image(mupost$pmean)

# plot edge weights
w <- as.vector(pmean)
o <- order(w)
blank2(w=3)
plot( w[o] , xlab="edge (sorted by weight)" , ylab="posterior weight" , col="white" )
pci <- apply( mpost$p_tie_out , 2:3 , PI , prob=0.95 )
phi <- as.vector(pci[2,,])[o]
plo <- as.vector(pci[1,,])[o]
for ( i in 1:length(phi) ) lines( c(i,i) , c(phi[i],plo[i]) )

# Plot the individual effects
blank2(w=2)
par(mfrow=c(1,2))
plot( dat$v[,1] , mpost$v_est[,1] )
plot( datu$v[,2] , mupost$v_est[,2] )

# Get some typical network descriptives
networks <- list(
    m_graph =  dat$y_true,
    m_graph_est = p_tie_out,
    mu_graph = datu$y_true,
    mu_graph_est = u_p_tie_out
        )

descriptive_table <- function(networks) {
	output <- setNames(data.frame(matrix(ncol = 7, nrow = 0)), c(
                                                               "density", 
                                                               "reciprocity", 
                                                               "transitivity", 
                                                               "indegree_mean", 
                                                               "outdegree_mean",
                                                               "indegree", 
                                                               "outdegree"
                                                               )
                                                                )

  	nets <- graph_from_adjacency_matrix(as.matrix(networks), mode="directed")
    den <- round(edge_density(nets), 4)                 # get the network density
    rec <- round(reciprocity(nets), 4)                  # general tendency towards reciprocity
    tra <- round(transitivity(nets), 4)                 # general tendency towards transitivity
    ind <- mean(degree(nets, mode = "in"))              # indegree for plotting
    oud <- mean(degree(nets, mode = "out"))             # outdegree for plotting
    rin <- range(degree(nets, mode="in"))               # range of indegree
    rim <- paste(rin[1], rin[2], sep = " - ")
    rou <- range(degree(nets, mode="out"))              # range of outdegree
    rov <- paste(rou[1], rou[2], sep = " - ")
    output <- rbind(output, data.frame( 
                                       density = den, 
                                       reciprocity = rec, 
                                       transitivity = tra, 
                                       indegree_mean = ind,
                                       outdegree_mean = oud,
                                       indegree = rim, 
                                       outdegree = rov,  stringsAsFactors=FALSE))
  output
}

sapply(networks, descriptive_table)

# Get the rate of false positives/negatives & plot the networks 
network_plots <- function(true_network, inferred_network, type, groups, inferred_groups, initial_groups) {


# plot true network with inferred network

confusion <- table( true_network , inferred_network )
print(confusion)
# calculate Jaccard similarity between networks 
a <- confusion["1","1"] + confusion["1","0"]
b <- confusion["1","1"] + confusion["0","1"]
ab_ind <- a*b/sum(confusion)
jaccard <- ab_ind / (a + b - ab_ind)
    cat("Expected Jaccard under independence with these marginals =", jaccard, "\n")
    cat("Observed Jaccard = ")
         print(confusion["1","1"]/(confusion["1","1"] + confusion["0","1"] + confusion["1","0"]))

# Set coordinates
lx <- layout_nicely(graph_from_adjacency_matrix(as.matrix(true_network)))

if ( type == "true_inferred") {

	#blank2(w=2)
	par(mfrow=c(1,2))
		plot(graph_from_adjacency_matrix(as.matrix(true_network)) ,  
			layout=layout_nicely , vertex.size=8 , edge.arrow.size=0.55 , edge.curved=0.35 , 
			vertex.color= groups , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 ,
			vertex.label=NA , main="truth" )
		plot(graph_from_adjacency_matrix(as.matrix(inferred_network)) ,  
			layout=layout_nicely , vertex.size=8 , edge.arrow.size=0.55 , edge.curved=0.35 , 
			vertex.color= groups , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 ,
	 		vertex.label=NA , main= "posterior mean" )
	}

if ( type == "pie" ) {

gmean <- apply( inferred_groups , 2:3 , mean )
    gest <- sapply( 1:nrow(gmean) , function(i) which.max( gmean[i,] ) )
    # prob of each group as list, for vertex.shape="pie"
    glist <- lapply( 1:nrow(gmean) , function(i) as.integer( 10L*round( gmean[i,] , 1 ) ) )
        # calculate accuracy
        cat("Group accuracy")
            print(table( gest[initial_groups<0]==groups[initial_groups<0] ))

#blank2(w=2)	
	par(mfrow=c(1,2))
		plot(graph_from_adjacency_matrix(as.matrix(true_network)) , layout=lx , vertex.size=8 , edge.arrow.size=0.55 ,
 			edge.curved=0.35 , vertex.color= groups , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 ,
 			vertex.label=NA , main="truth" )

		plot(graph_from_adjacency_matrix(as.matrix(inferred_network)), layout=lx , vertex.size=8 , edge.arrow.size=0.55 , edge.curved=0.35 , 
			vertex.shape="pie" , vertex.pie=glist , vertex.pie.color=list(c(1,2,3,4)) , edge.color=gray(0.5,0.5) ,
			asp=0.9 , margin = -0.05 , vertex.label=NA, main="posterior mean"  )

	}

if ( type == "elly" ) {

    gmean <- apply( inferred_groups , 2:3 , mean )
    gest <- sapply( 1:nrow(gmean) , function(i) which.max( gmean[i,] ) )
    # prob of each group as list, for vertex.shape="pie"
    glist <- lapply( 1:nrow(gmean) , function(i) as.integer( 10L*round( gmean[i,] , 1 ) ) )
        # calculate accuracy
        cat("Group accuracy")
            print(table( gest[initial_groups<0]==groups[initial_groups<0] ))

# blank2(w=4)
par(mfrow=c(1,4))
plot(graph_from_adjacency_matrix(as.matrix(true_network)) , layout=lx , vertex.size=8 , edge.arrow.size=0.55 ,
    edge.curved=0.35 , vertex.color= groups , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 ,
      vertex.label=NA , main="truth" )
for ( i in 1:length(unique(groups))) 
    plot( graph_from_adjacency_matrix(as.matrix(inferred_network)) , layout=lx , vertex.size=8 , edge.arrow.size=0.55 ,
        edge.curved=0.35 , edge.color=gray(0.5,0.5) , asp=0.9 , margin = -0.05 , vertex.label=NA , 
            vertex.color=gray(1-gmean[,i]) , main=concat("clique ",i) )
    } 
}

# Plot the different options that we had 

# true_inferred
network_plots(type = "true_inferred" , true_network = datu$y_true , inferred_network = u_p_tie_out ,
    groups = dat$group)

# pie
network_plots(type = "pie" , true_network = datu$y_true , inferred_network = u_p_tie_out ,
    groups = dat$group , inferred_groups = mupost$p_group , initial_groups = datu$group)

# elly
network_plots(type = "elly" , true_network = datu$y_true , inferred_network = u_p_tie_out , 
    groups = dat$group , inferred_groups = mupost$p_group , initial_groups = datu$group)
