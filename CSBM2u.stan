// (covariate) stochastic block model with partly UNOBSERVED groups
data{
    int N_id;
    int N_groups;
    int N_gifts;
    int group[N_id];            // -1 indicates unobserved
    int s[N_id,N_id,2];
    int g[N_id,N_id,N_gifts];
}
parameters{
    matrix[N_groups,N_groups] B;
    real r_mean;
    real r1;
    real g_mean;
    real g1;
    simplex[N_groups] pg; // vector of base rates of individuals being in each group
}
model{
    
    g_mean ~ normal(0,1);
    r_mean ~ normal(0,1);
    r1 ~ normal(0,1);
    g1 ~ normal(0,1);

    // priors for B
    for ( i in 1:N_groups )
        for ( j in 1:N_groups ) {
            if ( i==j ) {
                B[i,j] ~ normal(0,1); // transfers more likely within groups
            } else {
                B[i,j] ~ normal(-4,0.5); // transfers less likely between groups
            }
        }

    // group membership prior
    // assumes exclusive membership
    pg ~ dirichlet( [ 4 , 2 , 2 ]' );

    // likelihood
    for ( i in 1:N_id ) {
        for ( j in 1:N_id ) {
            if ( i != j ) {
                int k; // temp counting variable for mixture terms

                // both groups observed
                if ( group[i]>0 && group[j]>0 ) {
                    vector[2] terms;
                    // consider each possible state of true tie and compute prob of data
                    for ( tie in 0:1 ) {
                        terms[tie+1] = 
                            // prob i says i helps j
                            bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                            // prob i did help j on N_gift occasions
                            bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                            // prob j says i helps j
                            bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                    }
                    target += log_mix( inv_logit(B[group[i],group[j]]) , terms[2] , terms[1] );
                }//both observed

                // group A observed
                // marginalize over 2*N_groups terms, because 2 terms (tie=1,tie=0) for each possible group that B could be in
                if ( group[i]>0 && group[j]<0 ) {
                    vector[ 2 * N_groups ] terms;
                    k = 1;
                    for ( gB in 1:N_groups ) {
                        int tie;
                        tie = 0;
                        terms[k] = 
                            log(pg[gB]) + // prob B in group
                            log1m_inv_logit(B[group[i],gB]) + // prob of tie=0
                            // prob i says i helps j
                            bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                            // prob i did help j on N_gift occasions
                            bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                            // prob j says i helps j
                            bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                        k = k + 1;
                        tie = 1;
                        terms[k] = 
                            log(pg[gB]) + // prob B in group
                            log_inv_logit(B[group[i],gB]) + // prob of tie=1
                            // prob i says i helps j
                            bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                            // prob i did help j on N_gift occasions
                            bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                            // prob j says i helps j
                            bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                        k = k + 1;
                    }//gB
                    target += log_sum_exp( terms );
                }// A observed

                // group B observed
                if ( group[i]<0 && group[j]>0 ) {
                    vector[ 2 * N_groups ] terms;
                    k = 1;
                    for ( gA in 1:N_groups ) {
                        int tie;
                        tie = 0;
                        terms[k] = 
                            log(pg[gA]) + // prob A in group
                            log1m_inv_logit(B[gA,group[j]]) + // prob of tie=0
                            // prob i says i helps j
                            bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                            // prob i did help j on N_gift occasions
                            bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                            // prob j says i helps j
                            bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                        k = k + 1;
                        tie = 1;
                        terms[k] = 
                            log(pg[gA]) + // prob A in group
                            log_inv_logit(B[gA,group[j]]) + // prob of tie=1
                            // prob i says i helps j
                            bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                            // prob i did help j on N_gift occasions
                            bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                            // prob j says i helps j
                            bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                        k = k + 1;
                    }//gB
                    target += log_sum_exp( terms );
                }// B observed

                // neither group observed
                // so need N_groups*N_groups*2 terms
                if ( group[i]<0 && group[j]<0 ) {
                    vector[ 2 * N_groups * N_groups ] terms;
                    k = 1;
                    for ( gA in 1:N_groups ) {
                        for ( gB in 1:N_groups ) {
                            int tie;
                            tie = 0;
                            terms[k] = 
                                log(pg[gA]) + log(pg[gB]) + // prob groups
                                log1m_inv_logit(B[gA,gB]) + // prob of tie=0
                                // prob i says i helps j
                                bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                                // prob i did help j on N_gift occasions
                                bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                                // prob j says i helps j
                                bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                            k = k + 1;
                            tie = 1;
                            terms[k] = 
                                log(pg[gA]) + log(pg[gB]) + // prob groups
                                log_inv_logit(B[gA,gB]) + // prob of tie=1
                                // prob i says i helps j
                                bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                                // prob i did help j on N_gift occasions
                                bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                                // prob j says i helps j
                                bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                            k = k + 1;
                        }//gB
                    }//gA
                    target += log_sum_exp( terms );
                }//neither observed 

            }//i != j
        }//j
    }//i

}
