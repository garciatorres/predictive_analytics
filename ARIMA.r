data <- btc_multivariate
Y <- data[,9] #the label is the closing price (column 9)
X <- data[-9] #feature space without the label
X <- X[-1] #drop the date column

# X[i] will predict Y[i+1]
X <- X[-nrow(X),] #drop the last row of the regressors
Y = Y[-1,] # drop first row of labels

# train and test data (31 days)
train_X <- head(X,nrow(Y)-31)
test_X <- tail(X,31)
train_Y <- head(Y,nrow(Y)-31)
test_Y <- tail(Y,31)

#normalize feature space
test_XN <- do.call(cbind.data.frame, Map(cbind , test_X, lapply(test_X, function(x) round((x-min(x))/(max(x)-min(x)), 1))))
train_XN <- do.call(cbind.data.frame, Map(cbind , train_X, lapply(train_X, function(x) round((x-min(x))/(max(x)-min(x)), 1))))

even_indexes<-seq(2,ncol(X)*2,2)
train_XN <- data.frame(x=train_XN[,even_indexes])
is.nan.data.frame <- function(x) do.call(cbind, lapply(x, is.nan))
train_XN[is.nan(train_XN)] <- 0

# feature names
covariates <- colnames(test_X)
covariatesN <- colnames(train_XN)

# Feature selection
# NOTE: the auto.arima function of R package does not find suitable ARIMA models for 75 features ahead
cov11 <- head(covariates,11)
cov11N <- head(covariatesN,11)
crypto_close <- tail(head(covariates,75),16) #cryptocurrencies closing prices
crypto <- c(cov11,crypto_close)
cov75 <- head(covariates,75)
cov75N <- head(covariatesN,75)
cov117 <- head(covariates,117)
cov198 <- head(covariates,198)
cov213 <- head(covariates,213)

# procedure based on this source: https://www.otexts.org/fpp/9/1
# automatic ARIMA model without regressors
start_time <- Sys.time()
ARIMA.model <- auto.arima(train_Y, stepwise=FALSE, approximation=FALSE)
train_time <- Sys.time() - start_time
ARIMA.forecast <- forecast(ARIMA.model, h=31)
plot(ARIMA.forecast, xlab="1614 days (from May 2nd, 2013)", ylab="BTC/USD", lwd = 2)

# automatic ARIMA model with regressors (11 features)

start_time <- Sys.time()
DR.model <- auto.arima(train_Y,xreg=train_X[,cov11])
train_time <- Sys.time() - start_time
DR.forecast <- forecast(DR.model, xreg = test_X[,cov11],h=31)
plot(DR.forecast, xlab="1614 days (from May 2nd, 2013)", ylab="BTC/USD", lwd = 2)

# plot and compare Dynamic Regression forecasts
DR.forecast.df <- as.data.frame(DR.forecast)
DR.forecast.df.Y <- DR.forecast.df[,2]
DR.forecast.df.Y.ts <- zoo(ts(DR.forecast.df.Y,start=c(2017,9),frequency=365.25))
plot(DR.forecast.df.Y.ts)
test_Y.ts <- zoo(ts(test_Y,start=c(2017,9),frequency=365.25))

# plot in two panels
plot.zoo(cbind(test_Y.ts, DR.forecast.df.Y.ts))

# Overplotted
plot.zoo(cbind(test_Y.ts,DR.forecast.df.Y.ts), plot.type = "single", col=c("blue", "orange"), xlab="31 days (from September 1st, 2017)", ylab="BTC/USD", main = "Real BTC price vs Predicted BTC price", lwd = 2)
legend(x= "topright", y=0.92, legend=c("real","predictions"), col=c("blue", "orange"), lty=c(1,1))

# plot and compare ARIMA forecasts
ARIMA.forecast.df <- as.data.frame(ARIMA.forecast)
ARIMA.forecast.df.Y <- ARIMA.forecast.df[,2]
ARIMA.forecast.df.Y.ts <- zoo(ts(ARIMA.forecast.df.Y,start=c(2017,9),frequency=365.25))
plot(ARIMA.forecast.df.Y.ts)

# plot in two panels
plot.zoo(cbind(test_Y.ts, ARIMA.forecast.df.Y.ts))

# Overplotted
plot.zoo(cbind(test_Y.ts,ARIMA.forecast.df.Y.ts), plot.type = "single", col=c("blue", "orange"), xlab="31 days (from September 1st, 2017)", ylab="BTC/USD", main = "Real BTC price vs Predicted BTC price", lwd = 2)
legend(x= "topright", y=0.92, legend=c("real","predictions"), col=c("blue", "orange"), lty=c(1,1))

# model accuracy
accuracy(ARIMA.forecast, test_Y.ts)
accuracy(DR.forecast, test_Y.ts)
