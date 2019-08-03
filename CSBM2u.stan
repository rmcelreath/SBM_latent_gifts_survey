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
    real<lower=0> r1;
    real g_mean;
    real<lower=0> g1;
    simplex[N_groups] pg; // vector of base rates of individuals being in each group
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
                B[i,j] ~ normal(-4,0.5); // transfers less likely between groups
            }
        }

    // group membership prior
    // assumes exclusive membership
    pg ~ dirichlet( [ 2 , 2 , 2 ]' );

    // likelihood
    for ( i in 1:N_id ) {
        for ( j in 1:N_id ) {
            if ( i != j ) {
                int k; // temp counting variable for mixture terms

                // both groups observed
                if ( group[i]>0 && group[j]>0 ) {
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
                    group[i] ~ categorical( pg );
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
generated quantities{
    // compute posterior prob of each network tie
    // compute posterior prob of each group membership
    matrix[N_id,N_id] p_tie_out;
    matrix[N_id,N_groups] p_group;

    // likelihood
    for ( i in 1:N_id ) {
        vector[N_groups] termsg; // vector of probability terms for calculating prob of i being in each group --- accumulate as loop over j
        p_group[i,] = rep_vector(0,N_groups)';
        termsg = rep_vector(0,N_groups);

        for ( j in 1:N_id ) {
            p_tie_out[i,j] = 0;

            if ( i != j ) {
                int k; // temp counting variable for mixture terms

                // both groups observed
                if ( group[i]>0 && group[j]>0 ) {
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
                        terms1[k1] = terms[k];
                        k = k + 1;
                        k1 = k1 + 1;
                        // group prob terms - Pr(s,g|gA) [marginal of true tie]
                        termsg[gA] = termsg[gA] + log_sum_exp( terms[(k-2):(k-1)] );
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
                            terms1[k1] = terms[k];
                            k = k + 1;
                            k1 = k1 + 1;
                        }//gB
                        // group prob terms
                        termsg[gA] = termsg[gA] + log_sum_exp( terms[(k-2*N_groups):(k-1)] );
                    }//gA
                    //target += log_sum_exp( terms );
                    // need Pr(y|tie) / Pr(y)
                    p_tie_out[i,j] = exp(
                            log_sum_exp(terms1) - log_sum_exp(terms)
                        );
                }//neither observed 

            } //i != j else

        }//j

        // group for i not observed, so compute post prob now using accculated termsg
        if ( group[i]<0 ) {
            real Z;
            real NUM;
            Z = log_sum_exp( termsg ); // denominator
            for ( j in 1:N_groups ) {
                NUM = termsg[j];
                p_group[i,j] = exp( NUM - Z );
            }
            if ( i==3 ) {
                print(termsg);
                print(Z);
                print(p_group[i,]);
            }
        }

    }//i

}
