# This script reads in a csv file of parameters and output runtime values and creates polynomial interpolants
# Originally created by Hengrui Luo of the GPTune team

df = read.csv('./csv/diffusion-cvode-5-128-newton-gmres.csv')
for(i in 1:5){
  df1 = df[df$maxord==i & df$runtime<100.,2:6]
  pairs(df1)
  y = df1$runtime
  #Throw outliers out
  data = y
  quartiles <- quantile(data, probs=c(.25, .75), na.rm = FALSE)
  IQR <- IQR(data)
  
  Lower <- quartiles[1] - 1.5*IQR
  Upper <- quartiles[2] + 1.5*IQR 
  
  data_no_outlier <- subset(data, data > Lower & data < Upper)
  y=data_no_outlier
  idx = which(data > Lower & data < Upper)
  
  x1 = df1$nonlin_conv_coef[idx]
  x2 = df1$max_conv_fails[idx]#/50
  x3 = df1$maxl[idx]#/500
  x4 = df1$epslin[idx]
  if (i==1){
    fit.train = lm(y ~ poly(x1, x2, x3, x4, degree=2, raw=TRUE)) # is equivalent to  #y ~ x1 + x2 + I(x1^2) + I(x2^2) + x1:x2
    print(fit.train$coefficients)
    #fit.train$coefficients[abs(fit.train$coefficients)<=0.0001]<-0
  }else{
    #fit.train = lm(log(y) ~ poly(x1, x2, x3, x4, degree=2, raw=TRUE))
    fit.train = lm(log(y) ~ x1+x2+x3+x4)
    fit.train$coefficients[abs(fit.train$coefficients)<=0.01]<-0
    #fit.train$coefficients[2]<-c()
  }
  
  testData = df1[,1:4]
  fit.test = predict(fit.train, newdata=testData)
  cat('maxord=',i,' ')
  message( mean(abs(fit.test-y)) )
  print( as.numeric(fit.train$coefficients) )
}

