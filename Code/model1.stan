data {

    int<lower = 0> N; // number of observations
    int<lower = 0> K; // number of covariates
    matrix[N, K]   a; // sensitive variables
    //real           transcript[N]; // transcript
    int            view[N]; // view
    //real           rating[N]; // rating
    vector[N]           transcript; // transcript
    // int                 view[N]; // view
    vector[N]           rating; // rating
}

transformed data {
  
    vector[K] zero_K;
    vector[K] one_K;


    zero_K = rep_vector(0,K);
    one_K = rep_vector(1,K);

    


}



parameters {

    //The unknown variable treated as parameter
    vector[N] u;

    //prior probability
    real transcript0;
    real view0;

    //effect of u on transcript, view and rating
    real eta_u_transcript;
    real eta_u_view;
    real eta_u_rating;

    //effect of protected attiributes on transcript, view and rating
    vector[K] eta_a_transcript;
    vector[K] eta_a_view;
    vector[K] eta_a_rating;
    
    //effect of transcript on view and rating
    real eta_transcript_view;
    real eta_transcript_rating;

    //effect of view on rating
    real eta_view_rating;
  
    real<lower=0> sigma_transcript_sq;
    real<lower=0> sigma_rating_sq;
}


transformed parameters  {
    // Population standard deviation (a positive real number)
    real<lower=0> sigma_transcript;
    real<lower=0> sigma_rating;
    // Standard deviation (derived from variance)
    sigma_transcript = sqrt(sigma_transcript_sq);
    sigma_rating = sqrt(sigma_rating_sq);
}




model {
    
    // don't have data about this
    u ~ normal(0, 1);

    //prior sampling
    transcript0      ~ normal(0, 1);
    view0     ~ normal(0, 1);

    //effect of u 
    eta_u_transcript ~ normal(0, 1);
    eta_u_view ~ normal(0, 1);
    eta_u_rating ~ normal(0, 1);

    //effect of protected attribute
    eta_a_transcript ~ normal(zero_K, one_K);
    eta_a_view ~ normal(zero_K, one_K);
    eta_a_rating ~ normal(zero_K, one_K);

    //effect of transcript on view and rating
    eta_transcript_view ~ normal(0, 1);
    eta_transcript_rating ~ normal(0, 1);

    //effect of view on rating
    eta_view_rating ~ normal(0, 1);

    sigma_transcript_sq ~ inv_gamma(1, 1);
    sigma_rating_sq ~ inv_gamma(1, 1);

    // have data about these
    
    //view ~ poisson(exp(  transcript*eta_transcript_view));
    transcript ~ normal(transcript0 + eta_u_transcript * u + a * eta_a_transcript, sigma_transcript);
    view ~ poisson(exp(view0 + eta_u_view * u +  transcript * eta_transcript_view + a*eta_a_view));//transcript * eta_transcript_view +

    for (i in 1:N){   
      rating[i] ~ normal(eta_u_rating * u[i] + a[i] * eta_a_rating+transcript[i]* eta_transcript_rating+ view[i] * eta_view_rating, sigma_rating);
    }
}






