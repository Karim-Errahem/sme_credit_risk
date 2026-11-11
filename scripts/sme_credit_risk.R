library(readxl)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(PRROC)
library(ggplot2)
library(reshape2)


file_path <- "C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/data/9. sme_credit_default_1000x20_en.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Transformation de la variable cible en facteur avec des noms valides ("No", "Yes")
# 0 -> "No" (Non défaillant), 1 -> "Yes" (Défaillant)
df$default_12m <- factor(df$default_12m, levels = c(0, 1), labels = c("No", "Yes"))

# Identification et conversion des variables catégorielles
cat_cols <- c("loan_purpose", "industry_sector", "geographic_area", "legal_form", "application_channel")
df[cat_cols] <- lapply(df[cat_cols], as.factor)

# 3. Partitionnement des données (Train 75% / Test 25%)

set.seed(42)
trainIndex <- createDataPartition(df$default_12m, p = 0.75, list = FALSE)
train_data <- df[trainIndex, ]
test_data  <- df[-trainIndex, ]

# Configuration du Cross-Validation (5-Fold CV axé sur le ROC)
control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)


# 4. Entraînement des Modèles


# --- 1. Logistic Regression ---
set.seed(42)
model_glm <- train(
  default_12m ~ ., 
  data = train_data, 
  method = "glm", 
  family = "binomial",
  metric = "ROC",
  trControl = control
)

# --- 2. Random Forest ---
set.seed(42)
model_rf <- train(
  default_12m ~ ., 
  data = train_data, 
  method = "rf", 
  metric = "ROC",
  trControl = control,
  tuneLength = 3
)

# --- 3. Support Vector Machine (SVM) ---
set.seed(42)
model_svm <- train(
  default_12m ~ ., 
  data = train_data, 
  method = "svmRadial", 
  metric = "ROC",
  trControl = control,
  tuneLength = 3,
  preProcess = c("center", "scale")
)

# --- 4. Gradient Boosting (GBM) ---
set.seed(42)
model_gbm <- train(
  default_12m ~ ., 
  data = train_data, 
  method = "gbm", 
  metric = "ROC",
  trControl = control,
  verbose = FALSE
)


# 5. Évaluation sur le jeu de test (Held-out Test Set)

models_list <- list(
  "Logistic Regression" = model_glm,
  "Random Forest"        = model_rf,
  "SVM"                 = model_svm,
  "Gradient Boosting"   = model_gbm
)

results <- data.frame(
  Model = character(), 
  ROC_AUC = numeric(), 
  PR_AUC = numeric(), 
  Brier_Score = numeric(),
  stringsAsFactors = FALSE
)

for (name in names(models_list)) {
  model <- models_list[[name]]
  
  # Prédiction des probabilités pour la classe positive "Yes"
  preds_prob <- predict(model, newdata = test_data, type = "prob")[, "Yes"]
  actuals_numeric <- ifelse(test_data$default_12m == "Yes", 1, 0)
  
  # 1. Calcul ROC-AUC
  roc_res <- roc(actuals_numeric, preds_prob, quiet = TRUE)
  roc_auc_val <- as.numeric(auc(roc_res))
  
  # 2. Calcul PR-AUC (Fonctionne maintenant grâce à library(PRROC))
  pr_res <- pr.curve(
    scores.class0 = preds_prob[actuals_numeric == 1], 
    scores.class1 = preds_prob[actuals_numeric == 0], 
    curve = FALSE
  )
  pr_auc_val <- pr_res$auc.integral
  
  # 3. Calcul Brier Score
  brier_val <- mean((preds_prob - actuals_numeric)^2)
  
  # Stockage des résultats
  results <- rbind(results, data.frame(
    Model = name, 
    ROC_AUC = round(roc_auc_val, 4), 
    PR_AUC = round(pr_auc_val, 4), 
    Brier_Score = round(brier_val, 4)
  ))
}

# Affichage des résultats triés par ROC-AUC décroissant
print(results[order(-results$ROC_AUC), ])






# 6. Préparation du dossier d'enregistrement

output_dir <- "C:/Users/HP/Desktop/MASTER DATA/Statistical Machine Learning/SME_Credit_Risk/figures"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


# 7. Graphique 1 : Comparaison des Métriques (ROC-AUC & PR-AUC)

# Transformation des résultats pour ggplot2


results_melted <- melt(results[, c("Model", "ROC_AUC", "PR_AUC")], id.vars = "Model")

p1 <- ggplot(results_melted, aes(x = reorder(Model, value), y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("ROC_AUC" = "#2b5c8f", "PR_AUC" = "#d95f02"), 
                    labels = c("ROC-AUC", "PR-AUC")) +
  theme_minimal(base_size = 12) +
  labs(title = "Model Performance Comparison",
       subtitle = "Evaluation on independent test set",
       x = "Models",
       y = "Score",
       fill = "Metric") +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", hjust = 0.5))

ggsave(filename = file.path(output_dir, "1_metrics_comparison.png"), plot = p1, width = 8, height = 5, dpi = 300)


# 8. Graphique 2 : Courbes ROC (Gradient Boosting vs Random Forest vs Logistic)

png(filename = file.path(output_dir, "2_roc_curves.png"), width = 2400, height = 1800, res = 300)

# Tracer la première courbe (Gradient Boosting)
roc_gbm <- roc(actuals_numeric <- ifelse(test_data$default_12m == "Yes", 1, 0),
               predict(model_gbm, newdata = test_data, type = "prob")[, "Yes"], quiet = TRUE)
plot(roc_gbm, col = "#2b5c8f", lwd = 3, main = "ROC Curves of Models")

# Ajouter les autres modèles
roc_rf  <- roc(actuals_numeric, predict(model_rf, newdata = test_data, type = "prob")[, "Yes"], quiet = TRUE)
roc_svm <- roc(actuals_numeric, predict(model_svm, newdata = test_data, type = "prob")[, "Yes"], quiet = TRUE)
roc_glm <- roc(actuals_numeric, predict(model_glm, newdata = test_data, type = "prob")[, "Yes"], quiet = TRUE)

lines(roc_rf, col = "#27ae60", lwd = 2)
lines(roc_svm, col = "#e67e22", lwd = 2)
lines(roc_glm, col = "#e74c3c", lwd = 2, lty = 2)

legend("bottomright", 
       legend = c("Gradient Boosting (AUC = 0.877)", 
                  "Random Forest (AUC = 0.873)", 
                  "SVM (AUC = 0.695)", 
                  "Logistic Regression (AUC = 0.612)"),
       col = c("#2b5c8f", "#27ae60", "#e67e22", "#e74c3c"), 
       lwd = 2, lty = c(1,1,1,2), cex = 0.8)

dev.off()


# 9. Graphique 3 : Brier Score (Calibration)

p3 <- ggplot(results, aes(x = reorder(Model, -Brier_Score), y = Brier_Score, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 12) +
  labs(title = "Brier Score Comparison (Calibration)",
       subtitle = "Lower score indicates better probability estimation",
       x = "Models",
       y = "Brier Score") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave(filename = file.path(output_dir, "3_brier_scores.png"), plot = p3, width = 7, height = 4.5, dpi = 300)


# 10. Graphique 4 : Importance des Variables (Gradient Boosting)

var_imp <- summary(model_gbm$finalModel, plotit = FALSE)

p4 <- ggplot(var_imp[1:10, ], aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_bar(stat = "identity", fill = "#2b5c8f", width = 0.7) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  labs(title = "Top 10 Most Influential Variables",
       subtitle = "Based on Gradient Boosting model",
       x = "Variables",
       y = "Relative Importance (%)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave(filename = file.path(output_dir, "4_feature_importance.png"), plot = p4, width = 8, height = 5, dpi = 300)

cat("\n[OK] All plots have been generated and saved in:\n", output_dir, "\n")