library(tidyverse)
library(ranger)
library(pdp)

source("lib/multiplot.R")


xf0 = readRDS("./models/xf0.rds")
xf1 = readRDS("./models/xf1.rds")

accord = read_csv("./data/accord/accord_cut.csv")
accord$accord_trial = 1
sprint = read_csv("./data/sprint/sprint_cut.csv")
sprint$accord_trial = 0
data = select(rbind(sprint, accord), -c(X1, cvd, t_cvds, INTENSIVE))
data$hisp = as.integer(data$hisp)

cols = tolower(c(colnames(data)))
features = sprintf("x%d", seq(0,16))
colnames(data) = features

plot_importance = function(xf0, xf1, cols) {

  importances = (importance(xf0) + importance(xf1)) / 2

  ggplot(data = tibble(col = cols, imp = importances), aes(x=col, y=imp)) +
    geom_bar(stat = "identity", width=0.75) +
    labs(y = "Relative importance",
         x = "Covariate") +
    coord_flip() +
    theme_light()
}

plot_pdp = function(xf0, xf1, col_idx) {

  subsample = sample_n(data, 100)
  col = sprintf("x%d", col_idx)
  pdp_0 = as_tibble(partial(xf0, train = subsample, pred.var = col))
  pdp_1 = as_tibble(partial(xf1, train = subsample, pred.var = col))
  pdp = tibble(val = pdp_0[,1][[1]], yhat = -(pdp_0$yhat + pdp_1$yhat) / 2)
  ggplot(pdp, aes(x = val, y = yhat)) +
    geom_point() +
    theme_light() +
    ylim(-0.05, 0.1) +
    labs(x = cols[col_idx + 1], y = "Predicted ARR")
}

plot_importance(xf0, xf1, cols)

pdps = c()

for (i in 0:(length(cols) - 1)) {
  pdps[[i + 1]] = plot_pdp(xf0, xf1, i)
}

multiplot(pdps[[1]], pdps[[2]], pdps[[3]], pdps[[4]], pdps[[5]], pdps[[6]],
          pdps[[7]], pdps[[8]], pdps[[9]], pdps[[10]], pdps[[11]], pdps[[12]],
          pdps[[13]], pdps[[14]], pdps[[15]], pdps[[16]], pdps[[17]], cols = 4)

