train <- fromJSON("/Users/rrmkumar/Downloads/Cooking/train.json")
test <- fromJSON("/Users/rrmkumar/Downloads/Cooking/test.json")

test$cuisine <- NA

combi <- rbind(train, test)

library(tm)
library(NLP)
library(ggplot2)

#NLP AND TEXT MINING - Pre Processing

corpus <- Corpus(VectorSource(combi$ingredients))


# Convert text to lowercase
corpus <- tm_map(corpus, tolower)
corpus[[1]]

#emove Punctuation
corpus <- tm_map(corpus, removePunctuation)
corpus[[1]]

#Remove Stopwords

corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus[[1]]

#Remove Whitespaces

corpus <- tm_map(corpus, stripWhitespace)
corpus[[1]]

#erform Stemming
library('SnowballC')
corpus <- tm_map(corpus, stemDocument)
corpus[[1]]

corpus <- tm_map(corpus, PlainTextDocument)
#document matrix

frequencies <- DocumentTermMatrix(corpus) 
frequencies

#organizing frequency of terms
freq <- colSums(as.matrix(frequencies))
length(freq)

ord <- order(freq)
ord

#most and least frequent words
freq[head(ord)]
freq[tail(ord)]

#our table of 20 frequencies
head(table(freq),20)
tail(table(freq),20)


#remove sparse terms

sparse <- removeSparseTerms(frequencies, 1 - 3/nrow(frequencies))
dim(sparse)

#create a data frame for visualization
wf <- data.frame(word = names(freq), freq = freq)
head(wf)

#plot terms which appear atleast 10,000 times

library(ggplot2)
chart <- ggplot(subset(wf, freq >10000), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart


#find associated terms
findAssocs(frequencies, c('salt','oil'), corlimit=0.30)

#reate wordcloud
library(wordcloud)
library(RColorBrewer)
set.seed(142)
wordcloud(names(freq), freq, min.freq = 2500, scale = c(6, .1), colors = brewer.pal(4, "BuPu"))


#plot 5000 most used words
wordcloud(names(freq), freq, max.words = 5000, scale = c(6, .1), colors = brewer.pal(6, 'Dark2'))

#create sparse as data frame
newsparse <- as.data.frame(as.matrix(sparse))
dim(newsparse)
#check if all words are appropriate
colnames(newsparse) <- make.names(colnames(newsparse))

#check for the dominant dependent variable
table(train$cuisine)

#add cuisine
newsparse$cuisine <- as.factor(c(train$cuisine, rep('italian', nrow(test))))

#plit data 
mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]

library(xgboost)
library(Matrix)

# creating the matrix for training the model
ctrain <- xgb.DMatrix(Matrix(data.matrix(mytrain[,!colnames(mytrain) %in% c('cuisine')])), label = as.numeric(mytrain$cuisine)-1)

#advanced data set preparation
dtest <- xgb.DMatrix(Matrix(data.matrix(mytest[,!colnames(mytest) %in% c('cuisine')]))) 
watchlist <- list(train = ctrain, test = dtest)

#train multiclass model using softmax
#first model

xgbmodel <- xgboost(data = ctrain, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20, verbose = 1, watchlist = watchlist)

#second model
xgbmodel2 <- xgboost(data = ctrain, max.depth = 20, eta = 0.2, nrounds = 250, objective = "multi:softmax", num_class = 20, watchlist = watchlist)


#third model
xgbmodel3 <- xgboost(data = ctrain, max.depth = 25, gamma = 2, min_child_weight = 2, eta = 0.1, nround = 250, objective = "multi:softmax", num_class = 20, verbose = 2,watchlist = watchlist)



#predict 1
xgbmodel.predict <- predict(xgbmodel, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
xgbmodel.predict.text <- levels(mytrain$cuisine)[xgbmodel.predict + 1]

#predict 2
xgbmodel.predict2 <- predict(xgbmodel2, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')])) 
xgbmodel.predict2.text <- levels(mytrain$cuisine)[xgbmodel.predict2 + 1]

#predict 3
xgbmodel.predict3 <- predict(xgbmodel3, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')])) 
xgbmodel.predict3.text <- levels(mytrain$cuisine)[xgbmodel.predict3 + 1]

#data frame for predict 1
submit_match1 <- cbind(as.data.frame(test$id), as.data.frame(xgbmodel.predict.text))
colnames(submit_match1) <- c('id','cuisine')
submit_match1 <- data.table(submit_match1, key = 'id')

#data frame for predict 2
submit_match2 <- cbind(as.data.frame(test$id), as.data.frame(xgbmodel.predict2.text))
colnames(submit_match2) <- c('id','cuisine')
submit_match2 <- data.table(submit_match2, key = 'id')

#data frame for predict 3
submit_match3 <- cbind(as.data.frame(test$id), as.data.frame(xgbmodel.predict3.text))
colnames(submit_match3) <- c('id','cuisine')
submit_match3 <- data.table(submit_match3, key = 'id')


sum(diag(table(mytest$cuisine, xgbmodel.predict)))/nrow(mytest) 
sum(diag(table(mytest$cuisine, xgbmodel.predict2)))/nrow(mytest)
sum(diag(table(mytest$cuisine, xgbmodel.predict3)))/nrow(mytest)

submit_match3$cuisine2 <- submit_match2$cuisine 
submit_match3$cuisine1 <- submit_match1$cuisine


#function to find the maximum value row wise
Mode <- function(x) {
  u <- unique(x)
  u[which.max(tabulate(match(x, u)))]
}

x <- Mode(submit_match3[,c("cuisine","cuisine2","cuisine1")])
y <- apply(submit_match3,1,Mode)

final_submit <- data.frame(id= submit_match3$id, cuisine = y)
#view submission file
data.table(final_submit)

write.csv(final_submit,'/Users/rrmkumar/Downloads/Cooking/ensemble.csv',row.names = FALSE)
