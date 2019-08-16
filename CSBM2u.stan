// (covariate) stochastic block model with partly UNOBSERVED groups
functions{
    // probability of observed variables relevant to ij dyad
    // (1) s[i,j,1]: i's report of i->j
    // (2) s[j,i,2]: j's report of i->j
    // (3) g[i,j,]: i's gifts i->j
    real prob_sgij( int sij1, int sji2, int[] gij, int tie, vector alpha, vector beta ) {
        real y;
        y = 
            // prob i says helps j
            bernoulli_lpmf( sij1 | inv_logit( alpha[1] + beta[1]*tie ) ) + 
            // prob i did help j on N_gift occasions
            bernoulli_lpmf( gij | inv_logit( alpha[3] + beta[3]*tie ) ) + 
            // prob j says i helps j
            bernoulli_lpmf( sji2 | inv_logit( alpha[2] + beta[2]*tie ) );
        return(y);
    }
}
data{
    int N_id;
    int N_groups;               
    int N_gifts;
    int N_x;                    // number of outcomes to inform ties
    int group[N_id];            // -1 indicates unobserved
    int s[N_id,N_id,2];
    int g[N_id,N_id,N_gifts];
    vector[N_groups] pg_prior; // prior for dirichlet on blocks
}
parameters{
    matrix[N_groups,N_groups] B;
    vector[N_x] alpha;
    vector<lower=0>[N_x] beta;
    simplex[N_groups] pg; // vector of base rates of individuals being in each group
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
                B[i,j] ~ normal(-4,0.5); // transfers less likely between groups
            }
        }

    // group membership prior
    // assumes exclusive membership
    pg ~ dirichlet( pg_prior );

    // likelihood
    for ( i in 1:N_id ) {
        for ( j in 1:N_id ) {
            if ( i != j ) {
                int k; // temp counting variable for mixture terms
                real pij_logit;
                
                // both groups observed
                if ( group[i]>0 && group[j]>0 ) {
                    vector[2] terms;
                    int tie;
                    // consider each possible state of true tie and compute prob of data
                    pij_logit = B[group[i],group[j]] + v_id[i,1] + v_id[j,2];
                    tie = 0;
                    terms[1] = 
                        log1m_inv_logit( pij_logit ) + 
                        prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                    tie = 1;
                    terms[2] = 
                        log_inv_logit( pij_logit ) + 
                        prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                    target += log_sum_exp( terms );
                    // now prob of observed groups
                    group[i] ~ categorical( pg );
                    group[j] ~ categorical( pg );
                }//both observed

                // group A observed
                // marginalize over 2*N_groups terms, because 2 terms (tie=1,tie=0) for each possible group that B could be in
                if ( group[i]>0 && group[j]<0 ) {
                    vector[ 2 * N_groups ] terms;
                    k = 1;
                    for ( gB in 1:N_groups ) {
                        int tie;
                        pij_logit = B[group[i],gB] + v_id[i,1] + v_id[j,2];
                        tie = 0;
                        terms[k] = 
                            log(pg[gB]) + // prob B in group
                            log1m_inv_logit( pij_logit ) + // prob of tie=0
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        k = k + 1;
                        tie = 1;
                        terms[k] = 
                            log(pg[gB]) + // prob B in group
                            log_inv_logit( pij_logit ) + // prob of tie=1
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        k = k + 1;
                    }//gB
                    target += log_sum_exp( terms );
                    group[i] ~ categorical( pg );
                }// A observed

                // group B observed
                if ( group[i]<0 && group[j]>0 ) {
                    vector[ 2 * N_groups ] terms;
                    k = 1;
                    for ( gA in 1:N_groups ) {
                        int tie;
                        pij_logit = B[gA,group[j]] + v_id[i,1] + v_id[j,2];
                        tie = 0;
                        terms[k] = 
                            log(pg[gA]) + // prob A in group
                            log1m_inv_logit( pij_logit ) + // prob of tie=0
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        k = k + 1;
                        tie = 1;
                        terms[k] = 
                            log(pg[gA]) + // prob A in group
                            log_inv_logit( pij_logit ) + // prob of tie=1
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        k = k + 1;
                    }//gB
                    target += log_sum_exp( terms );
                    group[j] ~ categorical( pg );
                }// B observed

                // neither group observed
                // so need N_groups*N_groups*2 terms
                if ( group[i]<0 && group[j]<0 ) {
                    vector[ 2 * N_groups * N_groups ] terms;
                    k = 1;
                    for ( gA in 1:N_groups ) {
                        for ( gB in 1:N_groups ) {
                            int tie;
                            pij_logit = B[gA,gB] + v_id[i,1] + v_id[j,2];
                            tie = 0;
                            terms[k] = 
                                log(pg[gA]) + log(pg[gB]) + // prob groups
                                log1m_inv_logit( pij_logit ) + // prob of tie=0
                                prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                            k = k + 1;
                            tie = 1;
                            terms[k] = 
                                log(pg[gA]) + log(pg[gB]) + // prob groups
                                log_inv_logit( pij_logit ) + // prob of tie=1
                                prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                            k = k + 1;
                        }//gB
                    }//gA
                    target += log_sum_exp( terms );
                }//neither observed 

            }//i != j
        }//j
    }//i

}
generated quantities{
    // compute posterior prob of each network tie
    // compute posterior prob of each group membership
    matrix[N_id,N_id] p_tie_out;
    matrix[N_id,N_groups] p_group;

    // likelihood
    for ( i in 1:N_id ) {
        for ( j in 1:N_id ) {
            p_tie_out[i,j] = 0;

            if ( i != j ) {
                int k; // temp counting variable for mixture terms
                real pij_logit;

                // both groups observed
                if ( group[i]>0 && group[j]>0 ) {
                    vector[2] terms;
                    int tie;
                    // consider each possible state of true tie and compute prob of data
                    pij_logit = B[group[i],group[j]] + v_id[i,1] + v_id[j,2];
                    tie = 0;
                    terms[1] = 
                        log1m_inv_logit( pij_logit ) + 
                        prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                    tie = 1;
                    terms[2] = 
                        log_inv_logit( pij_logit ) + 
                        prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                    //target += log_sum_exp( terms );
                    p_tie_out[i,j] = exp(
                            terms[2] - log_sum_exp(terms)
                        );
                }//both observed

                // group A observed
                // marginalize over 2*N_groups terms, because 2 terms (tie=1,tie=0) for each possible group that B could be in
                if ( group[i]>0 && group[j]<0 ) {
                    vector[ 2 * N_groups ] terms;
                    vector[ N_groups ] terms1; // just the terms with tie=1
                    int k1;
                    k = 1;
                    k1 = 1;
                    for ( gB in 1:N_groups ) {
                        int tie;
                        pij_logit = B[group[i],gB] + v_id[i,1] + v_id[j,2];
                        tie = 0;
                        terms[k] = 
                            log(pg[gB]) + // prob B in group
                            log1m_inv_logit( pij_logit ) + // prob of tie=0
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        k = k + 1;
                        tie = 1;
                        terms[k] = 
                            log(pg[gB]) + // prob B in group
                            log_inv_logit( pij_logit ) + // prob of tie=1
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        terms1[k1] = terms[k];
                        k = k + 1;
                        k1 = k1 + 1;
                    }//gB
                    //target += log_sum_exp( terms );
                    // need Pr(y|tie) / Pr(y)
                    p_tie_out[i,j] = exp(
                            log_sum_exp(terms1) - log_sum_exp(terms)
                        );
                }// A observed

                // group B observed
                if ( group[i]<0 && group[j]>0 ) {
                    vector[ 2 * N_groups ] terms;
                    vector[ N_groups ] terms1;
                    int k1;
                    k = 1;
                    k1 = 1;
                    for ( gA in 1:N_groups ) {
                        int tie;
                        pij_logit = B[gA,group[j]] + v_id[i,1] + v_id[j,2];
                        tie = 0;
                        terms[k] = 
                            log(pg[gA]) + // prob A in group
                            log1m_inv_logit( pij_logit ) + // prob of tie=0
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        k = k + 1;
                        tie = 1;
                        terms[k] = 
                            log(pg[gA]) + // prob A in group
                            log_inv_logit( pij_logit ) + // prob of tie=1
                            prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                        terms1[k1] = terms[k];
                        k = k + 1;
                        k1 = k1 + 1;
                    }//gA
                    //target += log_sum_exp( terms );
                    // need Pr(y|tie) / Pr(y)
                    p_tie_out[i,j] = exp(
                            log_sum_exp(terms1) - log_sum_exp(terms)
                        );
                }// B observed

                // neither group observed
                // so need N_groups*N_groups*2 terms
                if ( group[i]<0 && group[j]<0 ) {
                    vector[ 2 * N_groups * N_groups ] terms;
                    vector[ N_groups * N_groups ] terms1;
                    int k1;
                    k = 1;
                    k1 = 1;
                    for ( gA in 1:N_groups ) {
                        for ( gB in 1:N_groups ) {
                            int tie;
                            pij_logit = B[gA,gB] + v_id[i,1] + v_id[j,2];
                            tie = 0;
                            terms[k] = 
                                log(pg[gA]) + log(pg[gB]) + // prob groups
                                log1m_inv_logit( pij_logit ) + // prob of tie=0
                                prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                            k = k + 1;
                            tie = 1;
                            terms[k] = 
                                log(pg[gA]) + log(pg[gB]) + // prob groups
                                log_inv_logit( pij_logit ) + // prob of tie=1
                                prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                            terms1[k1] = terms[k];
                            k = k + 1;
                            k1 = k1 + 1;
                        }//gB
                    }//gA
                    //target += log_sum_exp( terms );
                    // need Pr(y|tie) / Pr(y)
                    p_tie_out[i,j] = exp(
                            log_sum_exp(terms1) - log_sum_exp(terms)
                        );
                }//neither observed 

            } //i != j else

        }//j

    }//i

    // now compute prob of group membership
    // loop over individuals i, groups gA, and alters j
    for ( i in 1:N_id ) {
        vector[N_groups] termsg; // vector of probability terms for calculating prob of i being in each group --- accumulate as loop over j
        p_group[i,] = rep_vector(0,N_groups)';
        termsg = rep_vector(0,N_groups);

        if ( group[i] < 0 ) {
            //group not observed
            for ( gA in 1:N_groups ) {

                termsg[gA] = log(pg[gA]); // leading term

                for ( j in 1:N_id ) {
                    if ( i!=j ) {
                        real pij_logit;

                        // group B observed
                        // need prob of s,g|gA marginal of true tie
                        if ( group[i]<0 && group[j]>0 ) {
                            vector[ 2 ] terms;
                            int tie;
                            pij_logit = B[gA,group[j]] + v_id[i,1] + v_id[j,2];
                            tie = 0;
                            terms[1] = 
                                log1m_inv_logit( pij_logit ) + // prob of tie=0
                                prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                            tie = 1;
                            terms[2] = 
                                log_inv_logit( pij_logit ) + // prob of tie=1
                                prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                            // add Pr(g,s|gA) to termsg
                            termsg[gA] = termsg[gA] + log_sum_exp( terms );
                        }// B observed

                        // neither group observed
                        // need prob of s,g|gA marginal of tie and gB
                        if ( group[i]<0 && group[j]<0 ) {
                            vector[ 2 * N_groups ] terms;
                            int k;
                            k = 1;
                            for ( gB in 1:N_groups ) {
                                int tie;
                                pij_logit = B[gA,gB] + v_id[i,1] + v_id[j,2];
                                tie = 0;
                                terms[k] = 
                                    log(pg[gB]) + // prob gB
                                    log1m_inv_logit( pij_logit ) + // prob of tie=0
                                    prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                                k = k + 1;
                                tie = 1;
                                terms[k] = 
                                    log(pg[gB]) + // prob gB
                                    log_inv_logit( pij_logit ) + // prob of tie=1
                                    prob_sgij( s[i,j,1] , s[j,i,2] , g[i,j,] , tie , alpha , beta );
                                k = k + 1;
                            }//gB
                            termsg[gA] = termsg[gA] + log_sum_exp( terms );
                        }//neither observed

                    }//i!=j
                }//j

            }//gA

            // calculate Pr(gA|s,g)
            {
                real Z;
                Z = log_sum_exp( termsg ); // denominator
                for ( gA in 1:N_groups ) {
                    p_group[i,gA] = exp( termsg[gA] - Z );
                }
            }

        } else {
            // group observed - just insert observed
            p_group[i,group[i]] = 1;
        }
    }//i

}
