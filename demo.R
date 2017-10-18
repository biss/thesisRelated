corMat=matrix(c(1.0, 0.5, 0.25, 0.125, 0.5, 1.0, 0.5, 0.25, 0.25, 0.5, 1.0, 0.5, 0.125, 0.25, 0.5, 1.0),4,4)
cholMat=chol(corMat)
# create the matrix of random variables
set.seed(1000)
nValues=10000

matUniformAllSame=cbind(runif(nValues),runif(nValues),runif(nValues), runif(nValues))

# bind to a matrix
print("correlation Matrix")
print(corMat)
print("Cholesky Decomposition")
print (cholMat)

resultMatUniformAllSame=matUniformAllSame%*%cholMat
print("correlation matUniformAllSame")
print(cor(resultMatUniformAllSame))
