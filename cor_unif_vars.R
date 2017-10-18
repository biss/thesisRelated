# create the correlation matrix
n = 8
rou = 0.5
x = matrix(data=NA, nrow=n, ncol=n)

for(j in 1:n){
  for(i in 1:n){
    x[i,j] = `^`(rou, abs(i-j))
  }
}
print(x)
corMat = x

cholMat=chol(corMat)
# create the matrix of random variables
set.seed(1000)
nValues=10000

matUniformAllSame=cbind(runif(nValues),runif(nValues),runif(nValues),runif(nValues),runif(nValues),runif(nValues),runif(nValues),runif(nValues))
print("correlation Matrix")
print(corMat)
print("Cholesky Decomposition")
print (cholMat)

# test same uniform
resultMatUniformAllSame=matUniformAllSame%*%cholMat
print("correlation matUniformAllSame")
print(cor(resultMatUniformAllSame))

write.table(resultMatUniformAllSame, file="/home/biswajeet/Documents/PythonScript/thesisRelated/hey.csv", sep = ",", row.names = FALSE)

