library(GMMAT)
data(example)
attach(example)
model0 <- glmmkin(disease ~ age + sex, data = pheno, kins = GRM, id = "id",
       family = binomial(link = "logit"))
model0$theta
model0$coefficients
infile <- system.file("extdata", "geno.txt", package = "GMMAT")
outfile <- "glmm.score.text.testoutfile.txt"
glmm.score(model0, infile = infile, outfile = outfile, infile.nrow.skip = 5, 
            infile.ncol.skip = 3, infile.ncol.print = 1:3, 
            infile.header.print = c("SNP", "Allele1", "Allele2"))
