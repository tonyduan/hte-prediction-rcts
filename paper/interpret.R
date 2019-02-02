# ==
# Miscellaneous R code used for our paper. Run from the top-level directory.
#
# Contact: Tony Duan, tonyduan@cs.stanford.edu
# ==
library(tidyverse)
library(gridExtra)
library(gsubfn)
library(ranger)
library(pdp)

# ----------------------------------------------------------------------------
#   Interpretation of X-learner with Random Forests
# ----------------------------------------------------------------------------

xf0 = readRDS("./ckpts/xf0.rds")
xf1 = readRDS("./ckpts/xf1.rds")

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

grid.arrange(pdps[[1]], pdps[[2]], pdps[[3]], pdps[[4]], pdps[[5]], pdps[[6]],
             pdps[[7]], pdps[[8]], pdps[[9]], pdps[[10]], pdps[[11]],
             pdps[[12]], pdps[[13]], pdps[[14]], pdps[[15]], pdps[[16]],
             pdps[[17]], ncol = 4)

# ----------------------------------------------------------------------------
#   Interpretation of logistic regression
# ----------------------------------------------------------------------------

cvd = select(rbind(sprint, accord), cvd)
t_cvds = select(rbind(sprint, accord), t_cvds)
treat = select(rbind(sprint, accord), INTENSIVE)$INTENSIVE

add_interaction_terms = function(df, treat_var) {
  n_cols = ncol(df)
  col_name = paste("x", n_cols, sep = "")
  df = mutate(df, !!col_name := treat_var)
  for (i in 1:(ncol(df) - 1)) {
    col_name = paste("x", i + n_cols, sep = "")
    df = mutate(df, !!col_name := treat_var * df[,i][[1]])
  }
  return(df)
}

binarize_outcome = function(df, cvd, t_cvds, cens_time = 3 * 365.25) {
  outcomes = cvd & t_cvds < 3 * 365.25
  cens_var = !cvd & t_cvds < 3 * 365.25
  return(list(outcomes = outcomes[!cens_var], df = df[!cens_var,]))
}

data_normalized = as_tibble(scale(data))
data_normalized = add_interaction_terms(data_normalized, treat)
list[outcomes_binarized, data_binarized] = binarize_outcome(data_normalized,
                                                            cvd, t_cvds)

logreg = glm(outcomes_binarized ~ ., data = data_binarized, family = "binomial")

weights_df = tibble(col = c(cols, "treat", paste(cols, "interact", sep = "_")),
                    interact = c(rep(0, 18), rep(1, 17)),
                    sign = sign(coef(logreg)[2:length(coef(logreg))]),
                    weight = abs(coef(logreg)[2:length(coef(logreg))]))

weights_df %>% filter(interact == 1) %>% arrange(desc(weight))
weights_df %>% filter(interact == 0) %>% arrange(desc(weight))

ggplot(weights_df, aes(x = col, y = weight, fill = interact)) +
  geom_bar(stat = "identity", width = 0.75) +
  labs(x = "Relative weight", y = "Column") +
  coord_flip() + theme_light()

