# SME credit default - tout le projet dans un seul fichier
# Comment lancer : mettre le working directory dans le dossier SME_Credit_Risk
# (celui qui contient le dossier data/), puis faire source("sme_credit_risk.R")

library(ggplot2)
library(caret)        # createDataPartition, createFolds
library(pROC)         # roc, auc
library(randomForest)
library(gbm)
library(e1071)        # svm



# ---------------------------------------------------------------
# 1. Import des donnees
# ---------------------------------------------------------------
df <- read.csv("C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/data/9. sme_credit_default_1000x20_en.csv", stringsAsFactors = TRUE)
str(df)
sum(is.na(df))               # pas de valeurs manquantes
table(df$default_12m)        # 716 non-defaut, 284 defaut

# ---------------------------------------------------------------
# 2. Analyse exploratoire (simple)
# ---------------------------------------------------------------
df$status <- factor(df$default_12m, labels = c("No default", "Default"))

p1 <- ggplot(df, aes(status)) + geom_bar() + labs(title = "Defaut ou non", x = "", y = "Nombre")
ggsave("C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures/class_balance.png", p1, width = 5, height = 4)

p2 <- ggplot(df, aes(status, cash_buffer_months)) + geom_boxplot() +
  labs(title = "Cash buffer selon le defaut", x = "", y = "Mois de cash")
ggsave("C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures/boxplot_cash.png", p2, width = 5, height = 4)

# taux de defaut par categorie
print(aggregate(default_12m ~ loan_purpose, df, mean))
print(aggregate(default_12m ~ industry_sector, df, mean))
print(aggregate(default_12m ~ prior_payment_delays, df, mean))
df$status <- NULL

# ---------------------------------------------------------------
# 3. Train / test split (70% / 30%)
# ---------------------------------------------------------------
idx   <- createDataPartition(df$default_12m, p = 0.7, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]
ytr <- train$default_12m
yte <- test$default_12m

# matrice avec dummies pour logistique et SVM, puis standardisation
# avec la moyenne et l'ecart-type du TRAIN seulement
X   <- model.matrix(default_12m ~ ., df)[, -1]
Xtr <- X[idx, ]
Xte <- X[-idx, ]
mu  <- colMeans(Xtr)
sdv <- apply(Xtr, 2, sd)
Xtr <- scale(Xtr, center = mu, scale = sdv)
Xte <- scale(Xte, center = mu, scale = sdv)
# (petite simplification : dans la CV on garde la standardisation du train entier)

# ---------------------------------------------------------------
# 4. Fonctions utiles
# ---------------------------------------------------------------
# PR-AUC (average precision)
pr_auc <- function(y, p) {
  o <- order(p, decreasing = TRUE)
  y <- y[o]
  sum((cumsum(y) / seq_along(y))[y == 1]) / sum(y)
}
roc_auc <- function(y, p) as.numeric(auc(roc(y, p, quiet = TRUE)))

# 5-fold CV sur le train : on donne une fonction qui fait les predictions
folds <- createFolds(ytr, k = 5)
cv_auc <- function(predict_fun) {
  res <- c()
  for (f in folds) {
    tr <- setdiff(seq_along(ytr), f)
    p  <- predict_fun(tr, f)
    res <- c(res, roc_auc(ytr[f], p))
  }
  mean(res)
}

# ---------------------------------------------------------------
# 5. Reglage des hyperparametres (CV sur le train seulement)
# ---------------------------------------------------------------
# Random Forest : mtry
rf_grid <- c(4, 8, 12, 16)
rf_cv <- sapply(rf_grid, function(m) {
  cv_auc(function(tr, va) {
    mod <- randomForest(train[tr, names(train) != "default_12m"], factor(ytr[tr]), mtry = m)
    predict(mod, train[va, names(train) != "default_12m"], type = "prob")[, 2]
  })
})
best_mtry <- rf_grid[which.max(rf_cv)]
print(cbind(mtry = rf_grid, auc = rf_cv))

# Gradient Boosting : profondeur et nombre d'arbres
gb_grid <- expand.grid(depth = c(2, 3, 4), trees = c(300, 600))
gb_cv <- apply(gb_grid, 1, function(g) {
  cv_auc(function(tr, va) {
    mod <- gbm(default_12m ~ ., data = train[tr, ], distribution = "bernoulli",
               n.trees = g["trees"], interaction.depth = g["depth"],
               shrinkage = 0.01, n.minobsinnode = 10, verbose = FALSE)
    predict(mod, train[va, ], n.trees = g["trees"], type = "response")
  })
})
best_gb <- gb_grid[which.max(gb_cv), ]
print(cbind(gb_grid, auc = gb_cv))

# SVM : cost et gamma
svm_grid <- expand.grid(cost = c(1, 4, 16), gamma = c(0.005, 0.01, 0.03))
svm_cv <- apply(svm_grid, 1, function(g) {
  cv_auc(function(tr, va) {
    mod <- svm(Xtr[tr, ], factor(ytr[tr]), kernel = "radial",
               cost = g["cost"], gamma = g["gamma"], probability = TRUE)
    pr <- predict(mod, Xtr[va, ], probability = TRUE)
    attr(pr, "probabilities")[, "1"]
  })
})
best_svm <- svm_grid[which.max(svm_cv), ]
print(cbind(svm_grid, auc = svm_cv))

# ---------------------------------------------------------------
# 6. Modeles finaux (entraines sur tout le train)
# ---------------------------------------------------------------
# regression logistique (baseline)
logit <- glm(ytr ~ ., data = data.frame(Xtr), family = binomial)
p_logit <- predict(logit, data.frame(Xte), type = "response")

# Random Forest
rf <- randomForest(train[, names(train) != "default_12m"], factor(ytr), mtry = best_mtry)
p_rf <- predict(rf, test[, names(test) != "default_12m"], type = "prob")[, 2]

# Gradient Boosting
gb <- gbm(default_12m ~ ., data = train, distribution = "bernoulli",
          n.trees = best_gb$trees, interaction.depth = best_gb$depth,
          shrinkage = 0.01, n.minobsinnode = 10, verbose = FALSE)
p_gb <- predict(gb, test, n.trees = best_gb$trees, type = "response")

# SVM
sv <- svm(Xtr, factor(ytr), kernel = "radial", cost = best_svm$cost,
          gamma = best_svm$gamma, probability = TRUE)
p_svm <- attr(predict(sv, Xte, probability = TRUE), "probabilities")[, "1"]

# ---------------------------------------------------------------
# 7. Evaluation sur le test
# ---------------------------------------------------------------
preds <- list("Logistic" = p_logit, "Random Forest" = p_rf,
              "Gradient Boosting" = p_gb, "SVM" = p_svm)

results <- data.frame(
  model   = names(preds),
  ROC_AUC = sapply(preds, function(p) roc_auc(yte, p)),
  PR_AUC  = sapply(preds, function(p) pr_auc(yte, p)),
  Brier   = sapply(preds, function(p) mean((p - yte)^2)))
rownames(results) <- NULL
print(results, digits = 3)
write.csv(results, "C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures/test_results.csv", row.names = FALSE)

# courbes ROC
png("C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures/roc_curves.png", width = 900, height = 700, res = 130)
cols <- c("darkgreen", "orange", "purple", "red")
plot(roc(yte, preds[[1]], quiet = TRUE), col = cols[1], main = "ROC curves (test set)")
for (i in 2:4) plot(roc(yte, preds[[i]], quiet = TRUE), col = cols[i], add = TRUE)
legend("bottomright", legend = names(preds), col = cols, lwd = 2)
dev.off()

# courbe de calibration (5 groupes) pour le meilleur modele
best_name <- results$model[which.max(results$ROC_AUC)]
p_best <- preds[[best_name]]
grp <- cut(p_best, quantile(p_best, 0:5 / 5), include.lowest = TRUE)
calib <- data.frame(predit = tapply(p_best, grp, mean), observe = tapply(yte, grp, mean))
p3 <- ggplot(calib, aes(predit, observe)) + geom_abline(linetype = 2) +
  geom_point(size = 3) + geom_line() + xlim(0, 1) + ylim(0, 1) +
  labs(title = paste("Calibration -", best_name), x = "Probabilite predite", y = "Taux observe")
ggsave("C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures/calibration.png", p3, width = 5, height = 5)

# ---------------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------------
# odds ratios de la logistique
print(round(sort(exp(coef(logit)[-1]), decreasing = TRUE), 2))

# importance par permutation (on melange une variable et on regarde la baisse d'AUC)
perm_imp <- function(predict_fun, vars) {
  base <- roc_auc(yte, predict_fun(test))
  sapply(vars, function(v) {
    drop <- replicate(10, {
      d <- test
      d[[v]] <- sample(d[[v]])
      base - roc_auc(yte, predict_fun(d))
    })
    mean(drop)
  })
}
vars <- names(train)[names(train) != "default_12m"]
imp <- perm_imp(function(d) predict(gb, d, n.trees = best_gb$trees, type = "response"), vars)
imp <- sort(imp, decreasing = TRUE)
print(round(head(imp, 8), 3))

imp_df <- data.frame(variable = factor(names(imp), levels = rev(names(imp))), drop = imp)
p4 <- ggplot(head(imp_df, 10), aes(variable, drop)) + geom_col() + coord_flip() +
  labs(title = "Importance par permutation (Gradient Boosting)", x = "", y = "Baisse de l'AUC")
ggsave("C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures/importance.png", p4, width = 7, height = 5)

# ---------------------------------------------------------------
# 9. Petit test : une interaction suffit pour la logistique ?
# ---------------------------------------------------------------
# On regarde seulement la CV sur le train (le test n'est pas utilise ici)
cv_logit <- function(formula) {
  cv_auc(function(tr, va) {
    m <- glm(formula, data = train[tr, ], family = binomial)
    predict(m, train[va, ], type = "response")
  })
}
cat("Logistique sans interaction :",
    round(cv_logit(default_12m ~ .), 3), "\n")
cat("Logistique avec revenue_growth x inventory_change :",
    round(cv_logit(default_12m ~ . + revenue_growth_pct:inventory_change_pct), 3), "\n")
