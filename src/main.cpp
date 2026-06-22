/*  main.cpp  —  Credit Scoring & Portfolio Loss
 *
 *  Pipeline:
 *    1. Load Lending Club CSV
 *    2. Build feature matrix, standardize, train/test split (80/20)
 *    3. Train LDA, k-NN (k=15), Logistic Regression
 *    4. Evaluate: accuracy, precision, recall, F1, AUC
 *    5. Build credit portfolio from test set (PD = logistic prob)
 *    6. Monte Carlo portfolio loss (one-factor Gaussian copula)
 *    7. Compute VaR 95/99, CVaR, Basel II capital requirement
 *    8. Write all results to data/ CSVs for R analysis
 *
 *  Compile:
 *    g++ -O2 -std=c++17 -I../include main.cpp -o credit_model
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>

#include "matrix.h"
#include "data_loader.h"
#include "classifiers.h"
#include "portfolio_mc.h"
#include "results_writer.h"

int main() {
    std::cout << "==================================================\n";
    std::cout << "  Credit Scoring & Portfolio Loss — C++ Engine    \n";
    std::cout << "==================================================\n\n";

    // ── 1. Load data ──────────────────────────────────────────────────────────
    std::cout << "[1] Loading data...\n";
    auto records = load_csv("../data/lending_club.csv");
    std::cout << "    Loaded " << records.size() << " loan records\n";

    int total_def = 0;
    for (auto& r : records) total_def += r.label;
    std::cout << "    Defaults: " << total_def
              << " (" << std::fixed << std::setprecision(1)
              << 100.0*total_def/records.size() << "%)\n\n";

    // ── 2. Feature matrix + train/test split ──────────────────────────────────
    std::cout << "[2] Building features & splitting data...\n";
    Matrix X_full; std::vector<int> y_full;
    std::vector<double> mu, sigma;
    build_features(records, X_full, y_full, mu, sigma, true);

    Matrix X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(X_full, y_full, X_train, y_train, X_test, y_test, 0.20, 42);
    std::cout << "    Train: " << X_train.rows
              << "  Test: "    << X_test.rows
              << "  Features: "<< X_train.cols << "\n\n";

    // ── 3. Train classifiers ──────────────────────────────────────────────────
    std::cout << "[3] Training classifiers...\n";

    // LDA
    std::cout << "    LDA... " << std::flush;
    LDAClassifier lda;
    lda.fit(X_train, y_train);
    auto m_lda = lda.evaluate(X_test, y_test);
    std::cout << "AUC=" << std::setprecision(4) << m_lda.auc
              << "  F1=" << m_lda.f1 << "\n";

    // k-NN
    std::cout << "    k-NN (k=15)... " << std::flush;
    KNNClassifier knn(15);
    knn.fit(X_train, y_train);
    // Use asymmetric threshold (0.35) because false negatives are more costly
    auto m_knn = knn.evaluate(X_test, y_test, 0.35);
    std::cout << "AUC=" << m_knn.auc
              << "  F1=" << m_knn.f1 << "\n";

    // Logistic Regression
    std::cout << "    Logistic Regression... " << std::flush;
    LogisticRegression lr(0.05, 1e-3, 300, 64);
    lr.fit(X_train, y_train, 42);
    auto m_lr = lr.evaluate(X_test, y_test, 0.40);
    std::cout << "AUC=" << m_lr.auc
              << "  F1=" << m_lr.f1 << "\n\n";

    // ── 4. Write classifier metrics ───────────────────────────────────────────
    std::cout << "[4] Writing classifier results...\n";
    write_metrics_csv("../data/metrics.csv",
                      {"LDA","KNN","LogisticRegression"},
                      {m_lda, m_knn, m_lr});

    // Collect all predicted probabilities for ROC / calibration plots
    int n_test = X_test.rows;
    std::vector<double> proba_lda(n_test), proba_knn(n_test), proba_lr(n_test);
    for (int i = 0; i < n_test; ++i) {
        auto r = X_test.row(i);
        proba_lda[i] = lda.predict_proba(r);
        proba_knn[i] = knn.predict_proba(r);
        proba_lr[i]  = lr.predict_proba(r);
    }
    write_all_probas_csv("../data/probas.csv",
                          {"LDA","KNN","LogisticRegression"},
                          {proba_lda, proba_knn, proba_lr},
                          y_test);

    // Feature importance from logistic regression
    auto imp = lr.importance(FEATURE_NAMES);
    write_feature_importance("../data/feature_importance.csv", imp);

    std::cout << "\n    Top-5 features (LogReg):\n";
    for (int i = 0; i < 5 && i < (int)imp.size(); ++i)
        std::cout << "      " << i+1 << ". " << imp[i].first
                  << " (" << std::setprecision(4) << imp[i].second << ")\n";

    // ── 5. Build portfolio ────────────────────────────────────────────────────
    std::cout << "\n[5] Building credit portfolio from test set...\n";

    double LGD = 0.45;  // Basel II standard for unsecured retail (45%)
    std::vector<Loan> portfolio;
    for (int i = 0; i < n_test; ++i) {
        auto row = X_test.row(i);
        double pd  = lr.predict_proba(row);
        pd = std::max(0.001, std::min(0.999, pd));  // clip for Phi^{-1}
        // Recover original (unstandardized) loan amount
        double loan_amnt = row[0] * sigma[0] + mu[0];  // feature 0 = loan_amnt
        Loan l;
        l.pd        = pd;
        l.ead       = loan_amnt;
        l.lgd       = LGD;
        l.rho_asset = basel_rho(pd);
        portfolio.push_back(l);
    }

    double total_exposure = 0;
    double total_el = 0;
    for (auto& l : portfolio) {
        total_exposure += l.ead;
        total_el       += l.pd * l.lgd * l.ead;
    }
    std::cout << "    Loans in portfolio:  " << portfolio.size() << "\n";
    std::cout << "    Total exposure:      $" << std::fixed << std::setprecision(0)
              << total_exposure << "\n";
    std::cout << "    Expected loss (EL):  $" << total_el << "  ("
              << std::setprecision(2) << 100*total_el/total_exposure << "%)\n\n";

    // ── 6. Monte Carlo portfolio simulation ───────────────────────────────────
    std::cout << "[6] Running Monte Carlo (100,000 scenarios)...\n";
    CreditPortfolioMC mc(100000, 42);
    auto results = mc.simulate(portfolio);

    std::cout << "    EL  (portfolio %): " << std::setprecision(4)
              << 100*results.el   << "%\n";
    std::cout << "    UL  (std dev):     " << 100*results.ul   << "%\n";
    std::cout << "    VaR 95%:           " << 100*results.var_95 << "%\n";
    std::cout << "    VaR 99%:           " << 100*results.var_99 << "%\n";
    std::cout << "    CVaR 95%:          " << 100*results.cvar_95 << "%\n";
    std::cout << "    CVaR 99%:          " << 100*results.cvar_99 << "%\n";
    std::cout << "    Basel II capital:  " << 100*results.capital_req << "%\n\n";

    // ── 7. Grade-level PD analysis ────────────────────────────────────────────
    std::cout << "[7] Grade-level PD analysis...\n";
    std::vector<std::string> grade_names = {"A","B","C","D","E","F","G"};
    std::vector<double> grade_pd(7, 0), grade_dr(7, 0);
    std::vector<int>    grade_cnt(7, 0), grade_def(7, 0);

    for (int i = 0; i < n_test; ++i) {
        int g = (int)(X_test.at(i,2) * sigma[2] + mu[2]);  // unstandardize grade
        g = std::max(0, std::min(6, g));
        grade_pd[g]  += proba_lr[i];
        grade_dr[g]  += y_test[i];
        grade_cnt[g]++;
        if (y_test[i]) grade_def[g]++;
    }
    for (int g = 0; g < 7; ++g) {
        if (grade_cnt[g] > 0) {
            grade_pd[g] /= grade_cnt[g];
            grade_dr[g]  = (double)grade_def[g] / grade_cnt[g];
        }
    }
    write_grade_pd("../data/grade_pd.csv", grade_names, grade_pd, grade_dr);

    // ── 8. Write Monte Carlo results ──────────────────────────────────────────
    std::cout << "[8] Writing portfolio results...\n";
    write_portfolio_summary("../data/portfolio_summary.csv", results);
    write_loss_distribution("../data/loss_distribution.csv", results.loss_dist);

    // Write threshold analysis (precision/recall tradeoff)
    {
        std::ofstream f("../data/threshold_analysis.csv");
        f << "threshold,accuracy,precision,recall,f1\n";
        f << std::fixed << std::setprecision(6);
        for (double t = 0.10; t <= 0.90; t += 0.02) {
            auto m = lr.evaluate(X_test, y_test, t);
            f << t << "," << m.accuracy << "," << m.precision
              << "," << m.recall << "," << m.f1 << "\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "  All results written to data/\n";
    std::cout << "==================================================\n";
    return 0;
}
