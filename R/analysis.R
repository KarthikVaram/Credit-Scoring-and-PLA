# =============================================================================
# Credit Scoring & Portfolio Loss — R Analysis
# =============================================================================
# All plot labels use ASCII-safe text (no raw Unicode Greek letters)
# to ensure correct rendering on all PNG/Cairo devices.
# =============================================================================
library(ggplot2); library(dplyr); library(tidyr); library(scales)
library(gridExtra); library(grid)
options(bitmapType = "cairo")

theme_credit <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title    = element_text(face="bold", size=base_size+2, hjust=0),
      plot.subtitle = element_text(color="grey40", size=base_size-1, hjust=0),
      plot.caption  = element_text(color="grey50", size=9, hjust=1),
      axis.title    = element_text(size=base_size),
      axis.text     = element_text(size=base_size-2),
      legend.title  = element_text(size=base_size-1, face="bold"),
      legend.text   = element_text(size=base_size-2),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color="grey90"),
      strip.text    = element_text(face="bold"),
      plot.background = element_rect(fill="white", color=NA)
    )
}

PAL <- c("#2C7BB6","#D7191C","#1A9641","#F4A11D","#984EA3","#FF7F00","#4B4B4B","#A65628")

save_p <- function(p, f, w=9, h=6) {
  ggsave(file.path("../plots", f), p, width=w, height=h, dpi=180, bg="white",
         device=ragg::agg_png)
  cat("  Saved:", f, "\n")
}

cat("=== Credit Scoring R Analysis ===\n")

# ── Load raw data for EDA ─────────────────────────────────────────────────────
raw <- read.csv("../data/lending_club.csv")
cat("Loaded", nrow(raw), "records\n\n")

# =============================================================================
# PLOT 1: EDA — Default rate by grade
# =============================================================================
cat("[1] Default rate by grade\n")
grade_dr <- raw %>%
  group_by(grade) %>%
  summarise(n=n(), defaults=sum(default), dr=mean(default), .groups="drop") %>%
  mutate(grade=factor(grade, levels=c("A","B","C","D","E","F","G")))

p1 <- ggplot(grade_dr, aes(x=grade, y=dr*100, fill=grade)) +
  geom_col(alpha=0.85, width=0.65) +
  geom_text(aes(label=sprintf("%.1f%%", dr*100)), vjust=-0.4, size=4, fontface="bold") +
  scale_fill_brewer(palette="RdYlGn", direction=-1) +
  labs(title="Empirical Default Rate by Loan Grade",
       subtitle="Lending Club 10,000-loan portfolio | Grade A = lowest risk",
       x="Loan Grade", y="Default Rate (%)",
       caption="Source: Lending Club synthetic dataset (Lending Club schema)") +
  theme_credit() + theme(legend.position="none")
save_p(p1, "01_default_rate_by_grade.png")

# =============================================================================
# PLOT 2: EDA — FICO score distributions (default vs paid)
# =============================================================================
cat("[2] FICO score distribution\n")
raw <- raw %>%
  mutate(fico = (fico_range_low + fico_range_high) / 2,
         Status = ifelse(default==1, "Default", "Paid"))

p2 <- ggplot(raw, aes(x=fico, fill=Status)) +
  geom_histogram(bins=40, alpha=0.7, position="identity") +
  scale_fill_manual(values=c("Default"=PAL[2], "Paid"=PAL[1])) +
  labs(title="FICO Score Distribution by Loan Outcome",
       subtitle="Defaulters cluster at lower FICO scores — key discriminating feature",
       x="FICO Score (mid-range)", y="Count", fill="Outcome") +
  theme_credit() + theme(legend.position="top")
save_p(p2, "02_fico_distribution.png")

# =============================================================================
# PLOT 3: EDA — DTI and interest rate vs default
# =============================================================================
cat("[3] DTI and interest rate vs default\n")
p3a <- ggplot(raw %>% filter(dti < 40), aes(x=Status, y=dti, fill=Status)) +
  geom_violin(alpha=0.7, trim=TRUE) +
  geom_boxplot(width=0.15, outlier.shape=NA, fill="white", alpha=0.8) +
  scale_fill_manual(values=c("Default"=PAL[2], "Paid"=PAL[1])) +
  labs(title="Debt-to-Income Ratio (DTI)", x=NULL, y="DTI (%)") +
  theme_credit() + theme(legend.position="none")

p3b <- ggplot(raw, aes(x=Status, y=int_rate, fill=Status)) +
  geom_violin(alpha=0.7, trim=TRUE) +
  geom_boxplot(width=0.15, outlier.shape=NA, fill="white", alpha=0.8) +
  scale_fill_manual(values=c("Default"=PAL[2], "Paid"=PAL[1])) +
  labs(title="Interest Rate", x=NULL, y="Interest Rate (%)") +
  theme_credit() + theme(legend.position="none")

p3 <- grid.arrange(p3a, p3b, ncol=2,
  top=textGrob("Key Features: Distribution by Loan Outcome",
               gp=gpar(fontface="bold", fontsize=14)))
ggsave("../plots/03_key_features.png", p3, width=11, height=5.5, dpi=180, bg="white",
       device=ragg::agg_png)
cat("  Saved: 03_key_features.png\n")

# =============================================================================
# PLOT 4: EDA — Loan amount and annual income
# =============================================================================
cat("[4] Loan amount distribution\n")
p4 <- ggplot(raw, aes(x=loan_amnt/1000, fill=Status)) +
  geom_histogram(bins=35, alpha=0.7, position="identity") +
  scale_fill_manual(values=c("Default"=PAL[2], "Paid"=PAL[1])) +
  scale_x_continuous(labels=label_dollar(suffix="k")) +
  facet_wrap(~grade, ncol=4) +
  labs(title="Loan Amount Distribution by Grade and Outcome",
       subtitle="Higher grades show smaller loan sizes on average",
       x="Loan Amount ($k)", y="Count", fill="Outcome") +
  theme_credit() + theme(legend.position="top", strip.text=element_text(face="bold"))
save_p(p4, "04_loan_amount_by_grade.png", w=12, h=6)

# =============================================================================
# PLOT 5: ROC Curves — all three models
# =============================================================================
cat("[5] ROC curves\n")
probas <- read.csv("../data/probas.csv")

compute_roc <- function(scores, labels) {
  n <- length(scores)
  ord <- order(scores, decreasing=TRUE)
  scores <- scores[ord]; labels <- labels[ord]
  P <- sum(labels); N <- n - P
  tp <- 0; fp <- 0
  tpr <- c(0); fpr <- c(0)
  for (i in 1:n) {
    if (labels[i]==1) tp <- tp+1 else fp <- fp+1
    tpr <- c(tpr, tp/P); fpr <- c(fpr, fp/N)
  }
  data.frame(fpr=fpr, tpr=tpr)
}

roc_lda <- compute_roc(probas$LDA, probas$true_label) %>% mutate(model="LDA")
roc_knn <- compute_roc(probas$KNN, probas$true_label) %>% mutate(model="k-NN")
roc_lr  <- compute_roc(probas$LogisticRegression, probas$true_label) %>%
             mutate(model="Logistic Regression")
roc_all <- bind_rows(roc_lda, roc_knn, roc_lr)

# Compute AUCs from the data
auc_val <- function(roc) {
  sum(diff(roc$fpr) * (head(roc$tpr,-1) + tail(roc$tpr,-1))/2)
}
auc_lda <- round(auc_val(roc_lda), 4)
auc_knn <- round(auc_val(roc_knn), 4)
auc_lr  <- round(auc_val(roc_lr),  4)

p5 <- ggplot(roc_all, aes(x=fpr, y=tpr, color=model)) +
  geom_line(linewidth=1.1) +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="grey60") +
  scale_color_manual(
    values=c("LDA"=PAL[1], "k-NN"=PAL[3], "Logistic Regression"=PAL[2]),
    labels=c(sprintf("LDA  (AUC=%.4f)", auc_lda),
             sprintf("k-NN  (AUC=%.4f)", auc_knn),
             sprintf("Logistic Reg.  (AUC=%.4f)", auc_lr)),
    name="Model") +
  labs(title="ROC Curves: Classifier Comparison",
       subtitle="LDA and Logistic Regression achieve similar discrimination (AUC ~0.70)",
       x="False Positive Rate (1 - Specificity)",
       y="True Positive Rate (Sensitivity)",
       caption="Dashed diagonal = random classifier") +
  theme_credit() + theme(legend.position=c(0.75, 0.25))
save_p(p5, "05_roc_curves.png")

# =============================================================================
# PLOT 6: Feature importance (Logistic Regression weights)
# =============================================================================
cat("[6] Feature importance\n")
imp <- read.csv("../data/feature_importance.csv") %>%
  mutate(feature=reorder(feature, importance))

p6 <- ggplot(imp, aes(x=feature, y=importance, fill=importance)) +
  geom_col(width=0.7, alpha=0.9) +
  coord_flip() +
  scale_fill_gradient(low="#92C5DE", high="#2C7BB6") +
  labs(title="Feature Importance: Logistic Regression Coefficients",
       subtitle="Absolute value of standardized coefficients | grade and FICO dominate",
       x=NULL, y="|Coefficient| (standardized)",
       caption="Features standardized before training — magnitudes are comparable") +
  theme_credit() + theme(legend.position="none")
save_p(p6, "06_feature_importance.png", w=9, h=5.5)

# =============================================================================
# PLOT 7: Threshold analysis — Precision / Recall tradeoff
# =============================================================================
cat("[7] Precision-Recall tradeoff\n")
thresh <- read.csv("../data/threshold_analysis.csv")

p7 <- ggplot(thresh, aes(x=threshold)) +
  geom_line(aes(y=precision, color="Precision"), linewidth=1.1) +
  geom_line(aes(y=recall,    color="Recall"),    linewidth=1.1) +
  geom_line(aes(y=f1,        color="F1 Score"),  linewidth=1.1, linetype="dashed") +
  geom_vline(xintercept=0.40, linetype="dotted", color="grey50") +
  annotate("text", x=0.42, y=0.55, label="threshold=0.40\n(chosen)", size=3.5, color="grey40") +
  scale_color_manual(values=c("Precision"=PAL[1],"Recall"=PAL[2],"F1 Score"=PAL[3]),
                     name=NULL) +
  scale_x_continuous(labels=percent) +
  labs(title="Decision Threshold Analysis: Precision vs Recall Tradeoff",
       subtitle="Logistic Regression | Lower threshold = more defaults flagged (higher recall)",
       x="Classification Threshold", y="Metric Value",
       caption="False negatives (missed defaults) are more costly than false positives") +
  theme_credit() + theme(legend.position="top")
save_p(p7, "07_threshold_analysis.png")

# =============================================================================
# PLOT 8: Predicted PD distribution by grade
# =============================================================================
cat("[8] PD by grade\n")
grade_pd <- read.csv("../data/grade_pd.csv") %>%
  mutate(grade=factor(grade, levels=c("A","B","C","D","E","F","G"))) %>%
  pivot_longer(cols=c(avg_pd, empirical_dr),
               names_to="type", values_to="rate") %>%
  mutate(type=recode(type, "avg_pd"="Model PD (LogReg)", "empirical_dr"="Empirical Default Rate"))

p8 <- ggplot(grade_pd, aes(x=grade, y=rate*100, fill=type)) +
  geom_col(position="dodge", alpha=0.85, width=0.65) +
  scale_fill_manual(values=c("Model PD (LogReg)"=PAL[1], "Empirical Default Rate"=PAL[2]),
                    name=NULL) +
  labs(title="Model PD vs Empirical Default Rate by Grade",
       subtitle="Logistic Regression PD calibration across Lending Club loan grades",
       x="Loan Grade", y="Probability of Default (%)",
       caption="Well-calibrated model: model PD should track empirical DR closely") +
  theme_credit() + theme(legend.position="top")
save_p(p8, "08_pd_by_grade.png")

# =============================================================================
# PLOT 9: Monte Carlo loss distribution
# =============================================================================
cat("[9] Portfolio loss distribution\n")
losses <- read.csv("../data/loss_distribution.csv")
summ   <- read.csv("../data/portfolio_summary.csv")

get_val <- function(name) summ$value[summ$metric==name]
el    <- get_val("EL")
v95   <- get_val("VaR_95")
v99   <- get_val("VaR_99")
cv95  <- get_val("CVaR_95")
cap   <- get_val("Capital_Req")

p9 <- ggplot(losses, aes(x=loss_pct*100)) +
  geom_histogram(bins=100, fill=PAL[1], alpha=0.7, color=NA) +
  geom_vline(xintercept=el*100,  color=PAL[3], linewidth=1.2, linetype="solid") +
  geom_vline(xintercept=v95*100, color=PAL[4], linewidth=1.2, linetype="dashed") +
  geom_vline(xintercept=v99*100, color=PAL[2], linewidth=1.2, linetype="dashed") +
  geom_vline(xintercept=cv95*100,color="#984EA3", linewidth=1.0, linetype="dotted") +
  annotate("text", x=el*100+0.5,   y=Inf, label=sprintf("EL=%.1f%%",el*100),
           vjust=2, hjust=0, size=3.5, color=PAL[3]) +
  annotate("text", x=v95*100+0.5,  y=Inf, label=sprintf("VaR95=%.1f%%",v95*100),
           vjust=3.5, hjust=0, size=3.5, color=PAL[4]) +
  annotate("text", x=v99*100+0.5,  y=Inf, label=sprintf("VaR99=%.1f%%",v99*100),
           vjust=5, hjust=0, size=3.5, color=PAL[2]) +
  annotate("text", x=cv95*100+0.5, y=Inf, label=sprintf("CVaR95=%.1f%%",cv95*100),
           vjust=6.5, hjust=0, size=3.5, color="#984EA3") +
  labs(title="Monte Carlo Portfolio Loss Distribution",
       subtitle="One-factor Gaussian copula | 100,000 scenarios | LGD=45% (Basel II retail)",
       x="Portfolio Loss (%)", y="Frequency",
       caption="EL=Expected Loss | VaR=Value at Risk | CVaR=Conditional VaR (Expected Shortfall)") +
  theme_credit()
save_p(p9, "09_loss_distribution.png", w=10, h=6)

# =============================================================================
# PLOT 10: Basel II Capital Requirement vs PD curve
# =============================================================================
cat("[10] Basel II capital requirement curve\n")
pd_seq <- seq(0.001, 0.45, by=0.001)
lgd    <- 0.45

# Basel II rho and K formulas replicated in R
erlang_rho <- function(pd) {
  k <- 50
  f <- (1 - exp(-k*pd)) / (1 - exp(-k))
  0.12*f + 0.24*(1-f)
}
basel_capital_r <- function(pd, lgd) {
  rho    <- erlang_rho(pd)
  phi_pd <- qnorm(pd)
  phi999 <- qnorm(0.999)
  cond   <- pnorm((phi_pd + sqrt(rho)*phi999) / sqrt(1-rho))
  lgd * (cond - pd)
}
cap_df <- data.frame(pd=pd_seq) %>%
  mutate(K  = sapply(pd, function(p) basel_capital_r(p, lgd)),
         rho = sapply(pd, erlang_rho))

p10a <- ggplot(cap_df, aes(x=pd*100, y=K*100)) +
  geom_line(color=PAL[2], linewidth=1.3) +
  geom_vline(xintercept=el*100, linetype="dashed", color="grey50") +
  annotate("text", x=el*100+0.5, y=max(cap_df$K)*100*0.85,
           label=sprintf("Portfolio\nEL=%.1f%%", el*100), size=3.5, hjust=0, color="grey40") +
  labs(title="Basel II Capital Requirement K(PD)",
       subtitle="LGD=45% | IRB formula | confidence level = 99.9%",
       x="Probability of Default PD (%)", y="Capital Requirement K (% of EAD)") +
  theme_credit()

p10b <- ggplot(cap_df, aes(x=pd*100, y=rho)) +
  geom_line(color=PAL[1], linewidth=1.3) +
  labs(title="Asset Correlation rho(PD)",
       subtitle="Basel II IRB formula — higher PD -> lower correlation",
       x="Probability of Default PD (%)", y="Asset Correlation rho") +
  theme_credit()

p10 <- grid.arrange(p10a, p10b, ncol=2,
  top=textGrob("Basel II IRB: Capital Requirement and Asset Correlation",
               gp=gpar(fontface="bold", fontsize=14)))
ggsave("../plots/10_basel_capital.png", p10, width=12, height=5.5, dpi=180, bg="white",
       device=ragg::agg_png)
cat("  Saved: 10_basel_capital.png\n")

# =============================================================================
# PLOT 11: Model metrics comparison bar chart
# =============================================================================
cat("[11] Model metrics comparison\n")
metrics <- read.csv("../data/metrics.csv") %>%
  pivot_longer(cols=c(accuracy,precision,recall,f1,auc),
               names_to="metric", values_to="value") %>%
  mutate(model=factor(model, levels=c("LDA","KNN","LogisticRegression"),
                      labels=c("LDA","k-NN","Logistic Reg.")),
         metric=factor(metric, levels=c("accuracy","precision","recall","f1","auc"),
                       labels=c("Accuracy","Precision","Recall","F1","AUC")))

p11 <- ggplot(metrics, aes(x=metric, y=value, fill=model)) +
  geom_col(position="dodge", alpha=0.87, width=0.7) +
  geom_text(aes(label=sprintf("%.3f", value)),
            position=position_dodge(0.7), vjust=-0.4, size=3.2) +
  scale_fill_manual(values=c("LDA"=PAL[1],"k-NN"=PAL[3],"Logistic Reg."=PAL[2]),
                    name="Model") +
  scale_y_continuous(limits=c(0,1.1)) +
  labs(title="Classifier Performance Comparison",
       subtitle="Test set (n=2000) | LDA achieves highest recall; LogReg best AUC",
       x="Metric", y="Value") +
  theme_credit() + theme(legend.position="top")
save_p(p11, "11_model_comparison.png", w=9, h=5.5)

# =============================================================================
# PLOT 12: Probability calibration
# =============================================================================
cat("[12] Probability calibration\n")
probas_cal <- probas %>%
  mutate(bin=cut(LogisticRegression, breaks=seq(0,1,0.1), include.lowest=TRUE)) %>%
  group_by(bin) %>%
  summarise(mean_pred=mean(LogisticRegression), mean_actual=mean(true_label), n=n(),
            .groups="drop") %>%
  filter(!is.na(bin))

p12 <- ggplot(probas_cal, aes(x=mean_pred, y=mean_actual)) +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="grey60") +
  geom_point(aes(size=n), color=PAL[1], alpha=0.85) +
  geom_line(color=PAL[1], linewidth=0.8) +
  scale_size_continuous(name="Loans in bin", range=c(2,10)) +
  labs(title="Probability Calibration: Logistic Regression",
       subtitle="Mean predicted PD vs empirical default rate per decile",
       x="Mean Predicted PD", y="Empirical Default Rate",
       caption="Points on the diagonal = perfectly calibrated model") +
  theme_credit()
save_p(p12, "12_calibration.png", w=8, h=6)

cat("\n=== All", length(list.files("../plots")), "plots saved to plots/ ===\n")
