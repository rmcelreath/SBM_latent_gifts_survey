// (covariate) stochastic block model with observed groups
functions{
    // probability of observed variables relevant to ij dyad
    // (1) s[i,j,1]: i's report of i->j
    // (2) s[j,i,2]: j's report of i->j
    // (3) g[i,j,]: i's gifts i->j
    real prob_sgij( int sij1, int sji2, int[] gij, int tie, vector alpha, vector beta , real theta ) {
        real y;
        y = 
            // prob i says helps j
            bernoulli_lpmf( sij1 | inv_logit( alpha[1] + beta[1]*tie ) ) + 
            // prob j says i helps j
            bernoulli_lpmf( sji2 | inv_logit( (1-sij1)*alpha[2] + beta[2]*tie + sij1*theta ) );
        for ( k in 1:size(gij) ) {
            if ( gij[k] > -1 )
                // prob i did help j on N_gift occasions
                y = y + bernoulli_lpmf( gij[k] | inv_logit( alpha[3] + beta[3]*tie ) );
        }//k
        return(y);
    }
}
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
    real theta;
    // varying effects on individuals
    matrix[ 2 + N_x*2 ,N_id] z_id;
    vector<lower=0>[ 2 + N_x*2 ] sigma_id;
    cholesky_factor_corr[ 2 + N_x*2 ] L_Rho_id;
}
transformed parameters{
    matrix[N_id, 2 + N_x*2 ] v_id; 
    // 2 + N_x*2 effects
    // [1] g_i: general tendency to form out ties
    // [2] r_i: general tendency to receive in ties
    // [3+] individual alpha/beta adjustments
    v_id = (diag_pre_multiply(sigma_id,L_Rho_id) * z_id)';
}
model{
    
    alpha ~ normal(0,1);
    beta ~ normal(1,1);
    theta ~ normal(0,0.5);

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
                vector[N_x] alpha_id = alpha + v_id[i, 3:(2+N_x) ]';
                vector[N_x] beta_id = beta + v_id[i, (3+N_x):(2+2*N_x) ]';
                // consider each possible state of true tie and compute prob of data
                for ( tie in 0:1 ) {
                    terms[tie+1] = 
                        prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha_id , beta_id , theta );
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
                vector[N_x] alpha_id = alpha + v_id[i, 3:(2+N_x) ]';
                vector[N_x] beta_id = beta + v_id[i, (3+N_x):(2+2*N_x) ]';
                pij_logit = B[group[i],group[j]] + v_id[i,1] + v_id[j,2];
                // consider each possible state of true tie and compute prob of data
                tie = 0;
                terms[1] = 
                    log1m_inv_logit( pij_logit ) + 
                    prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha_id , beta_id , theta );
                tie = 1;
                terms[2] = 
                    log_inv_logit( pij_logit ) + 
                    prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha_id , beta_id , theta );
                p_tie_out[i,j] = exp(
                        terms[2] - log_sum_exp( terms )
                    );
            }
        }//j
    }//i

}

