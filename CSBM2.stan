// (covariate) stochastic block model with observed groups
data{
    int N_id;
    int N_groups;
    int N_gifts;
    int group[N_id];
    int s[N_id,N_id,2];
    int g[N_id,N_id,N_gifts];
}
parameters{
    matrix[N_groups,N_groups] B;
    real r_mean;
    real<lower=0> r1;
    real g_mean;
    real<lower=0> g1;
}
model{
    
    g_mean ~ normal(0,1);
    r_mean ~ normal(0,1);
    r1 ~ normal(1,1);
    g1 ~ normal(1,1);

    // priors for B
    for ( i in 1:N_groups )
        for ( j in 1:N_groups ) {
            if ( i==j ) {
                B[i,j] ~ normal(0,1); // transfers more likely within groups
            } else {
                B[i,j] ~ normal(-3,0.5); // transfers less likely between groups
            }
        }

    // likelihood
    for ( i in 1:N_id ) {
        for ( j in 1:N_id ) {
            if ( i != j ) {
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
            }
        }//j
    }//i

}
generated quantities{
    // compute posterior prob of each network tie
    matrix[N_id,N_id] p_tie_out;

    for ( i in 1:N_id ) {
        for ( j in 1:N_id ) {
            p_tie_out[i,j] = 0;
            if ( i != j ) {
                vector[2] terms;
                int tie;
                // consider each possible state of true tie and compute prob of data
                tie = 0;
                terms[1] = 
                    log1m_inv_logit(B[group[i],group[j]]) + 
                    // prob i says i helps j
                    bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                    // prob i did help j on N_gift occasions
                    bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                    // prob j says i helps j
                    bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                tie = 1;
                terms[2] = 
                    log_inv_logit(B[group[i],group[j]]) + 
                    // prob i says i helps j
                    bernoulli_lpmf( s[i,j,1] | inv_logit( r_mean + r1*tie ) ) + 
                    // prob i did help j on N_gift occasions
                    bernoulli_lpmf( g[i,j,] | inv_logit( g_mean + g1*tie ) ) + 
                    // prob j says i helps j
                    bernoulli_lpmf( s[j,i,2] | inv_logit( r_mean + r1*tie ) );
                //target += log_mix( inv_logit(B[group[i],group[j]]) , terms[2] , terms[1] );
                p_tie_out[i,j] = exp(
                        terms[2] - log_sum_exp( terms )
                    );
            }
        }//j
    }//i

}

