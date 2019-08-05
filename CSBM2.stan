// (covariate) stochastic block model with observed groups
data{
    int N_x;
    int N_id;
    int N_groups;
    int N_gifts;
    int group[N_id];
    int s[N_id,N_id,2];
    int g[N_id,N_id,N_gifts];
}
parameters{
    matrix[N_groups,N_groups] B;
    vector[N_x] alpha;
    vector<lower=0>[N_x] beta;
    // varying effects on individuals
    matrix[2,N_id] z_id;
    vector<lower=0>[2] sigma_id;
    cholesky_factor_corr[2] L_Rho_id;
}
transformed parameters{
    matrix[N_id,2] v_id;
    v_id = (diag_pre_multiply(sigma_id,L_Rho_id) * z_id)';
}
model{
    
    alpha ~ normal(0,1);
    beta ~ normal(1,1);

    to_vector(z_id) ~ normal(0,1);
    sigma_id ~ normal(0,1);
    L_Rho_id ~ lkj_corr_cholesky(4);

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
                real pij;
                // consider each possible state of true tie and compute prob of data
                for ( tie in 0:1 ) {
                    terms[tie+1] = 
                        // prob i says i helps j
                        bernoulli_lpmf( s[i,j,1] | inv_logit( alpha[1] + beta[1]*tie ) ) + 
                        // prob i did help j on N_gift occasions
                        bernoulli_lpmf( g[i,j,] | inv_logit( alpha[3] + beta[3]*tie ) ) + 
                        // prob j says i helps j
                        bernoulli_lpmf( s[j,i,2] | inv_logit( alpha[2] + beta[2]*tie ) );
                }
                pij = inv_logit( B[group[i],group[j]] + v_id[i,1] + v_id[j,2] );
                target += log_mix( pij , terms[2] , terms[1] );
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
                real pij_logit;
                int tie;
                pij_logit = B[group[i],group[j]] + v_id[i,1] + v_id[j,2];
                // consider each possible state of true tie and compute prob of data
                tie = 0;
                terms[1] = 
                    log1m_inv_logit( pij_logit ) + 
                    // prob i says i helps j
                    bernoulli_lpmf( s[i,j,1] | inv_logit( alpha[1] + beta[1]*tie ) ) + 
                    // prob i did help j on N_gift occasions
                    bernoulli_lpmf( g[i,j,] | inv_logit( alpha[3] + beta[3]*tie ) ) + 
                    // prob j says i helps j
                    bernoulli_lpmf( s[j,i,2] | inv_logit( alpha[2] + beta[2]*tie ) );
                tie = 1;
                terms[2] = 
                    log_inv_logit( pij_logit ) + 
                    // prob i says i helps j
                    bernoulli_lpmf( s[i,j,1] | inv_logit( alpha[1] + beta[1]*tie ) ) + 
                    // prob i did help j on N_gift occasions
                    bernoulli_lpmf( g[i,j,] | inv_logit( alpha[3] + beta[3]*tie ) ) + 
                    // prob j says i helps j
                    bernoulli_lpmf( s[j,i,2] | inv_logit( alpha[2] + beta[2]*tie ) );
                //target += log_mix( inv_logit(B[group[i],group[j]]) , terms[2] , terms[1] );
                p_tie_out[i,j] = exp(
                        terms[2] - log_sum_exp( terms )
                    );
            }
        }//j
    }//i

}

