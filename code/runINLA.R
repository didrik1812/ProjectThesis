#######################################################################
## Stefanie Muff, September 2023
##
## Helgeland house sparrow data analysis
## Make predictions using Machine Learning (deep learning)
## In addition, at the end we give code for the genomics-based animal model fitted with INLA
## !! This requires that you are using R version 4.2 or higher (ideally even 4.3) !!
#######################################################################


# Packages needed for the script to run:
#setwd("C:\\Users\\didri\\OneDrive - NTNU\\9.semester\\Prosjekt\\ProjectThesis\\code")

data_path = "~/../../../../work/didrikls/ProjectThesis/data/"
args <- commandArgs(trailingOnly = TRUE)
phenotype <- args[1] # say which phenotype we want to predict
typeOfModel <- args[2] #should be "gene" or "EG" do we only focus on pure genetics effect or ExG?

if (!require(nadiv)) {
    install.packages("nadiv", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(pedigree)) {
    install.packages("pedigree", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(MASS)) {
    install.packages("MASS", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(MCMCpack)) {
    install.packages("MCMCpack", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(data.table)) {
    install.packages("data.table", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(SMisc)) {
    install.packages("~\\ProjectThesis\\code\\SMisc.tar.gz", repos = NULL, type = "source")
}

if (!require(dplyr)) {
    install.packages("dplyr", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(lme4)) {
    install.packages("lme4", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}


if (!require(MCMCglmm)) {
    install.packages("MCMCglmm", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(feather)) {
    install.packages("feather", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if(!require(INLA)){
    install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
}



# Sys.setenv(NOT_CRAN = "true")
# install.packages("arrow")

library(nadiv)
library(pedigree)
library(MASS)
library(MCMCpack)
library(MCMCglmm)
# This is a self-made package that I send you to install locally:
library(SMisc)

library(feather)

library(dplyr)
library(INLA)

# library(keras)
# library(tensorflow)

## Old packages no longer needed (but kept as comments, in case we need them in the future...)
# library(MasterBayes)
# library(pedigreemm)
# library(bdsmatrix)
# library(irlba)
# library(RSpectra)
# library(dplyr)

# Data preparation helper script:
source("h_dataPrep.r")

# Some data wranging to ensure that the IDs in the data correspond to the IDs in the A and G-matrices (nothing to worry about):
# indicates that some IDs are missing:
d.map[3110:3125, ]
# from this we see the number of anmals
Nanimals <- 3116


# In the reduced pedigree only Nanimals out of the 3147 IDs are preset.
d.map$IDC <- 1:nrow(d.map)

d.morph$IDC <- d.map[match(d.morph$ringnr, d.map$ringnr), "IDC"]

### Prepare for use in INLA -
d.morph$IDC4 <- d.morph$IDC3 <- d.morph$IDC2 <- d.morph$IDC



d.morph <- filter(d.morph, !is.na(eval(as.symbol(phenotype))))
# Get CV-indexes
ringnr_train <- pull(arrow::read_feather(paste(data_path, "temp/ringnr_train.feather", sep="")), "ringnr")
ringnr_test <- pull(arrow::read_feather(paste(data_path, "temp/ringnr_test.feather", sep = "")),"ringnr")

# make test and train set
d.morph_train <-filter(d.morph, ringnr %in% ringnr_train)
d.morph_test <- filter(d.morph,ringnr %in% ringnr_test)

n_train <- dim(d.morph_train)[1]
n_test <- dim(d.morph_test)[1]
N <- n_train + n_test
pheno_test <- d.morph_test[,phenotype]

# However, INLA has no predict function, so have to fill the test-values with NAs and then merge it back into the train-set
d.morph_test[,phenotype] <- NA

d.morph_train <- union_all(d.morph_train, d.morph_test)


##################################################################
### Run INLA based on the GBLUP approach
###
### To this end, use the GRM (genetic relatedness matrix) in the animal model
### 
### !!! This is very slow - account for at least 20-30min waiting time before inla terminates !!!
##################################################################

# Relatedness matrix from Henrik (vanRaden method 1) where +0.01 was already added to diagnoal!
d.Gmatrix <- read.table(paste(data_path,"gghatvr3.triangle.g", sep= ""), header = F, sep=" ")

# keep only relatednesses that are relevant for animals in d.morph
#d.Gmatrix <- d.Gmatrix[d.Gmatrix[,1] %in% d.morph$ID & d.Gmatrix[,2] %in% d.morph$ID, ]

# G is a sparse matrix object. We can also verify that it is symmetric (sometimes some numerical problems lead to non-symmetry)
G <- sparseMatrix(i=d.Gmatrix[,1],j=d.Gmatrix[,2],x=d.Gmatrix[,3],symmetric=T) 
G[,] <- as.numeric(G[,])
isSymmetric(G)

# Again extract the rows and columns for the individuals in the data set that we analyse
GG <- G[d.map[1:3116,3],d.map[1:3116,3]]

# To ensure that the matrix is positive definite, we do a computational trick (proposed by vanRaden 2008, see https://doi.org/10.3168/jds.2007-0980 :)
AAA <- diag(dim(GG)[1])
GGG  <- GG*0.99 + 0.01*AAA # replace by Identity matrix

# Need to derive the inverse to give to INLA
Cmatrix <- solve(GGG)
if (!isSymmetric(Cmatrix)){
  Cmatrix <- forceSymmetric(Cmatrix)
}

## 
## INLA formula
## 
#Here we use body mass as the response, and some fixed and random effects:
formula.mass = eval(as.symbol(phenotype)) ~ sex + FGRM + month + age +   outer + other +
  f(hatchyear,model="iid",hyper=list(
    prec=list(initial=log(1), prior="pc.prec",param=c(1,0.05))
  ))+
  f(IDC,model="iid",hyper=list(
    prec=list(initial=log(1), prior="pc.prec",param=c(1,0.05))
  )) +
  f(IDC2,values=1:3116,model="generic0",
    Cmatrix=Cmatrix,
    constr = TRUE,
    hyper=list(
      # The priors are relevant, need to discuss
      prec=list(initial=log(0.5), prior="pc.prec",param=c(sqrt(2),0.05))
    ))

model1.mass = inla(formula=formula.mass, family="gaussian",
                   data=d.morph_train,
                   control.family=list(hyper = list(theta = list(initial=log(0.5),  prior="pc.prec",param=c(sqrt(2),0.05)))),
                   control.compute=list(dic=F, return.marginals=FALSE)
                   # control.compute=list(config = TRUE)
)
print("INLA DONE")
#marignals hyperpar
#summar.hyperpa
#summary.random$IDC2
if (typeOfModel == "EG"){
    preds <- model1.mass$summary.fitted.values$mean[(n_train+1):N]

}
else{
    preds <- model1.mass$summary.random$IDC2$mean[(n_train+1):N]
}
MSE <- mean((preds - pheno_test)^2)
corr <- cor(preds, pheno_test, method="pearson")

write_feather(data.frame(corr = corr, score = MSE),paste(data_path,"/temp/INLA_result.feather", sep = ""))