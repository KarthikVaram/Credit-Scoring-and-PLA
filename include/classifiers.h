#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <random>
#include "matrix.h"

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────
struct ClassMetrics {
    double accuracy, precision, recall, f1, auc;
    int tp, fp, tn, fn;
    std::vector<double> pred_proba;  // for ROC
};

inline ClassMetrics compute_metrics(const std::vector<int>& y_true,
                                     const std::vector<int>& y_pred,
                                     const std::vector<double>& proba) {
    ClassMetrics m{};
    m.pred_proba = proba;
    int n = y_true.size();
    for (int i = 0; i < n; ++i) {
        if      (y_true[i]==1 && y_pred[i]==1) ++m.tp;
        else if (y_true[i]==0 && y_pred[i]==1) ++m.fp;
        else if (y_true[i]==0 && y_pred[i]==0) ++m.tn;
        else                                    ++m.fn;
    }
    m.accuracy  = (double)(m.tp + m.tn) / n;
    m.precision = m.tp + m.fp > 0 ? (double)m.tp / (m.tp + m.fp) : 0;
    m.recall    = m.tp + m.fn > 0 ? (double)m.tp / (m.tp + m.fn) : 0;
    m.f1        = m.precision + m.recall > 0
                  ? 2 * m.precision * m.recall / (m.precision + m.recall) : 0;

    // AUC via trapezoidal rule on ROC
    std::vector<std::pair<double,int>> scored(n);
    for (int i = 0; i < n; ++i) scored[i] = {proba[i], y_true[i]};
    std::sort(scored.begin(), scored.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    double auc = 0, tpr_prev = 0, fpr_prev = 0;
    int P = m.tp + m.fn, N = m.fp + m.tn;
    int tp_cur = 0, fp_cur = 0;
    for (auto& [p, label] : scored) {
        if (label == 1) ++tp_cur; else ++fp_cur;
        double tpr = P > 0 ? (double)tp_cur / P : 0;
        double fpr = N > 0 ? (double)fp_cur / N : 0;
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0;
        tpr_prev = tpr; fpr_prev = fpr;
    }
    m.auc = auc;
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Linear Discriminant Analysis (LDA)
//    Fisher's LDA for binary classification.
//    Computes within-class scatter Sw, between-class scatter Sb,
//    finds projection direction w = Sw^{-1}(mu1 - mu0).
// ─────────────────────────────────────────────────────────────────────────────
class LDAClassifier {
public:
    std::vector<double> w_;      // projection vector
    double threshold_;           // decision boundary on projected axis
    std::vector<double> mu0_, mu1_;

    void fit(const Matrix& X, const std::vector<int>& y) {
        int n = X.rows, p = X.cols;

        // Class means
        mu0_.assign(p, 0); mu1_.assign(p, 0);
        int n0 = 0, n1 = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) {
                if (y[i] == 0) { mu0_[j] += X.at(i,j); ++n0; }
                else           { mu1_[j] += X.at(i,j); ++n1; }
            }
            if (y[i] == 0) { /* counted above */ } // avoid double count
        }
        // Fix double-counting p
        n0 /= p; n1 /= p;
        for (int j = 0; j < p; ++j) { mu0_[j] /= n0; mu1_[j] /= n1; }

        // Within-class scatter Sw
        Matrix Sw(p, p, 0.0);
        for (int i = 0; i < n; ++i) {
            auto& mu = (y[i]==0) ? mu0_ : mu1_;
            for (int a = 0; a < p; ++a)
                for (int b = 0; b < p; ++b)
                    Sw.at(a,b) += (X.at(i,a)-mu[a]) * (X.at(i,b)-mu[b]);
        }
        // Regularise slightly to handle near-singular Sw
        for (int j = 0; j < p; ++j) Sw.at(j,j) += 1e-4;

        Matrix Sw_inv = invert(Sw);
        auto diff = vec::sub(mu1_, mu0_);
        w_ = Sw_inv * diff;

        // Threshold: midpoint of projected class means
        double proj0 = vec::dot(w_, mu0_);
        double proj1 = vec::dot(w_, mu1_);
        threshold_   = (proj0 + proj1) / 2.0;
    }

    // Signed score (positive => default side)
    double score(const std::vector<double>& x) const {
        return vec::dot(w_, x) - threshold_;
    }

    // Sigmoid-mapped probability (approximation)
    double predict_proba(const std::vector<double>& x) const {
        double s = score(x);
        return 1.0 / (1.0 + std::exp(-s));
    }

    int predict(const std::vector<double>& x) const {
        return score(x) >= 0 ? 1 : 0;
    }

    ClassMetrics evaluate(const Matrix& X, const std::vector<int>& y) const {
        int n = X.rows;
        std::vector<int>    pred(n);
        std::vector<double> proba(n);
        for (int i = 0; i < n; ++i) {
            auto r = X.row(i);
            pred[i]  = predict(r);
            proba[i] = predict_proba(r);
        }
        return compute_metrics(y, pred, proba);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. k-Nearest Neighbours
//    Euclidean distance in standardized feature space.
//    P(default) = fraction of k neighbours that defaulted.
// ─────────────────────────────────────────────────────────────────────────────
class KNNClassifier {
public:
    int k_;
    Matrix X_train_;
    std::vector<int> y_train_;

    explicit KNNClassifier(int k = 15) : k_(k) {}

    void fit(const Matrix& X, const std::vector<int>& y) {
        X_train_ = X;
        y_train_ = y;
    }

    double predict_proba(const std::vector<double>& x) const {
        int n = X_train_.rows;
        std::vector<std::pair<double,int>> dists(n);
        for (int i = 0; i < n; ++i) {
            double d = 0;
            for (int j = 0; j < X_train_.cols; ++j) {
                double diff = x[j] - X_train_.at(i,j);
                d += diff * diff;
            }
            dists[i] = {d, y_train_[i]};
        }
        std::partial_sort(dists.begin(), dists.begin() + k_, dists.end());
        double votes = 0;
        for (int i = 0; i < k_; ++i) votes += dists[i].second;
        return votes / k_;
    }

    int predict(const std::vector<double>& x, double threshold = 0.5) const {
        return predict_proba(x) >= threshold ? 1 : 0;
    }

    ClassMetrics evaluate(const Matrix& X, const std::vector<int>& y,
                          double threshold = 0.5) const {
        int n = X.rows;
        std::vector<int>    pred(n);
        std::vector<double> proba(n);
        for (int i = 0; i < n; ++i) {
            auto r   = X.row(i);
            proba[i] = predict_proba(r);
            pred[i]  = proba[i] >= threshold ? 1 : 0;
        }
        return compute_metrics(y, pred, proba);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. Logistic Regression (mini-batch gradient descent + L2 regularisation)
//    P(Y=1|x) = sigmoid(w^T x + b)
//    Loss: cross-entropy + lambda/2 * ||w||^2
// ─────────────────────────────────────────────────────────────────────────────
class LogisticRegression {
public:
    std::vector<double> w_;
    double b_ = 0;
    double lr_, lambda_;
    int    epochs_, batch_size_;

    LogisticRegression(double lr=0.05, double lambda=1e-3,
                       int epochs=200, int batch_size=64)
        : lr_(lr), lambda_(lambda), epochs_(epochs), batch_size_(batch_size) {}

    static double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

    void fit(const Matrix& X, const std::vector<int>& y, unsigned seed=42) {
        int n = X.rows, p = X.cols;
        w_.assign(p, 0.0); b_ = 0.0;
        std::mt19937 rng(seed);

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        for (int ep = 0; ep < epochs_; ++ep) {
            std::shuffle(idx.begin(), idx.end(), rng);
            for (int start = 0; start < n; start += batch_size_) {
                int end = std::min(start + batch_size_, n);
                std::vector<double> dw(p, 0.0); double db = 0;
                for (int ii = start; ii < end; ++ii) {
                    int i = idx[ii];
                    double z = b_;
                    for (int j = 0; j < p; ++j) z += w_[j] * X.at(i,j);
                    double err = sigmoid(z) - y[i];
                    for (int j = 0; j < p; ++j) dw[j] += err * X.at(i,j);
                    db += err;
                }
                int bs = end - start;
                for (int j = 0; j < p; ++j)
                    w_[j] -= lr_ * (dw[j] / bs + lambda_ * w_[j]);
                b_ -= lr_ * db / bs;
            }
        }
    }

    double predict_proba(const std::vector<double>& x) const {
        double z = b_;
        for (size_t j = 0; j < w_.size(); ++j) z += w_[j] * x[j];
        return sigmoid(z);
    }

    int predict(const std::vector<double>& x, double threshold = 0.5) const {
        return predict_proba(x) >= threshold ? 1 : 0;
    }

    ClassMetrics evaluate(const Matrix& X, const std::vector<int>& y,
                          double threshold = 0.5) const {
        int n = X.rows;
        std::vector<int>    pred(n);
        std::vector<double> proba(n);
        for (int i = 0; i < n; ++i) {
            auto r   = X.row(i);
            proba[i] = predict_proba(r);
            pred[i]  = proba[i] >= threshold ? 1 : 0;
        }
        return compute_metrics(y, pred, proba);
    }

    // Feature importance (absolute weight magnitude)
    std::vector<std::pair<std::string,double>>
    importance(const std::vector<std::string>& names) const {
        std::vector<std::pair<std::string,double>> imp;
        for (size_t j = 0; j < w_.size(); ++j)
            imp.push_back({names[j], std::abs(w_[j])});
        std::sort(imp.begin(), imp.end(),
                  [](auto& a, auto& b){ return a.second > b.second; });
        return imp;
    }
};
