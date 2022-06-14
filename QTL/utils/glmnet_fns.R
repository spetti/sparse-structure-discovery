
glmnet_coeffs <- function(tG,tP,vG,vP){
    #library(Matrix)
    #library(glmnet)
    #run glmnet
    #mfit <- glmnet(tG, tP, family = "mgaussian", standardize.response = FALSE, alpha=1, trace.it = TRUE)
    mfit <- glmnet(tG, tP, family = "mgaussian", standardize.response = FALSE, alpha=1)
    
    #evaluate model on test set
    pred <- predict(mfit, newx = vG)
    res <- assess.glmnet(pred, newy = vP, family = "mgaussian")
    
    #let lam be value with minimum error on validation set
    lam=which.min(res$mae)[[1]]
    
    #each column is coeffs in an env
    n_envs <- dim(tP)[2]
    coeffs=sapply(c(1:n_envs), function(env){as.vector(mfit$beta[[env]][,lam])})
    #coeffs=sapply(c(1:5), function(env){as.matrix(mfit$beta[[env]])})
    
    #predictions on test set; will need this for considering merges
    t_pred <- predict(mfit, newx = tG)
    return (list(coeffs, t_pred[,,lam]))
}


glmnet_coeffs_all <- function(tG,tP,vG,vP){
    #library(Matrix)
    #library(glmnet)
    #run glmnet
    #mfit <- glmnet(tG, tP, family = "mgaussian", standardize.response = FALSE, alpha=1, trace.it = TRUE)
    mfit <- glmnet(tG, tP, family = "mgaussian", standardize.response = FALSE, alpha=1)
    
    #evaluate model on test set
    pred <- predict(mfit, newx = vG)
    res <- assess.glmnet(pred, newy = vP, family = "mgaussian")
    
    #each column is coeffs in an env
    n_envs <- dim(tP)[2]
    coeffs=sapply(c(1:n_envs), function(env){as.array(mfit$beta[[env]])},simplify = "array")
    
    #predictions on test set; will need this for considering merges
    t_pred <- predict(mfit, newx = tG)
    return (list(coeffs, t_pred, as.array(mfit$lambda)))
}