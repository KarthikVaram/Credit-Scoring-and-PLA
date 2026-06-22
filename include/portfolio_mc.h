#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Single loan in the portfolio
// ─────────────────────────────────────────────────────────────────────────────
struct Loan {
    double pd;          // Probability of Default (from classifier)
    double ead;         // Exposure at Default (= loan_amnt)
    double lgd;         // Loss Given Default (assumed constant per Basel II)
    double rho_asset;   // Asset correlation (Basel II formula, grade-dependent)
};

// ─────────────────────────────────────────────────────────────────────────────
// Basel II asset correlation (IRB formula)
//   rho(PD) = 0.12 * (1 - exp(-50*PD))/(1 - exp(-50))
//           + 0.24 * [1 - (1 - exp(-50*PD))/(1 - exp(-50))]
// ─────────────────────────────────────────────────────────────────────────────
inline double basel_rho(double pd) {
    double k = 50.0;
    double f = (1.0 - std::exp(-k * pd)) / (1.0 - std::exp(-k));
    return 0.12 * f + 0.24 * (1.0 - f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Basel II Capital Requirement K(PD) for LGD=1:
//   K = LGD * [Phi(Phi^{-1}(PD)/sqrt(1-rho) + sqrt(rho/(1-rho))*Phi^{-1}(0.999))
//              - PD] * (1 + (M-2.5)*b) / (1 - 1.5*b)
// We use simplified K without maturity adjustment for clarity.
// ─────────────────────────────────────────────────────────────────────────────

// Standard normal CDF
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Standard normal inverse CDF (rational approximation, Beasley-Springer-Moro)
inline double norm_inv(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03,-3.223964580411365e-01,
                                -2.400758277161838e+00,-2.549732539343734e+00,
                                 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = { 7.784695709041462e-03, 3.224671290700398e-01,
                                 2.445134137142996e+00, 3.754408661907416e+00};
    double q, r;
    if (p < 0.02425) {
        q = std::sqrt(-2 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else if (p <= 0.97575) {
        q = p - 0.5; r = q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    } else {
        q = std::sqrt(-2 * std::log(1-p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
}

inline double basel_capital(double pd, double lgd) {
    double rho = basel_rho(pd);
    double phi_pd  = norm_inv(pd);
    double phi_999 = norm_inv(0.999);
    double cond_pd = norm_cdf((phi_pd + std::sqrt(rho) * phi_999)
                              / std::sqrt(1.0 - rho));
    return lgd * (cond_pd - pd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Monte Carlo Portfolio Loss Simulation (one-factor Gaussian copula)
//
//  Each loan i has idiosyncratic factor eps_i ~ N(0,1) and the portfolio
//  shares a systematic factor Z ~ N(0,1):
//
//    X_i = sqrt(rho_i) * Z + sqrt(1-rho_i) * eps_i
//
//  Loan i defaults in this scenario if X_i < Phi^{-1}(PD_i)
//  Loss_i = EAD_i * LGD_i * 1{default}
//  Portfolio loss = sum_i Loss_i
// ─────────────────────────────────────────────────────────────────────────────
struct PortfolioResults {
    double el;           // Expected Loss
    double ul;           // Unexpected Loss (std dev)
    double var_95;       // VaR at 95%
    double var_99;       // VaR at 99%
    double cvar_95;      // CVaR (Expected Shortfall) at 95%
    double cvar_99;      // CVaR at 99%
    double capital_req;  // Basel II capital requirement (sum over loans)
    std::vector<double> loss_dist;  // full simulated distribution
};

class CreditPortfolioMC {
public:
    int n_sim_;
    unsigned seed_;

    explicit CreditPortfolioMC(int n_sim = 100000, unsigned seed = 42)
        : n_sim_(n_sim), seed_(seed) {}

    PortfolioResults simulate(const std::vector<Loan>& loans) const {
        std::mt19937_64 rng(seed_);
        std::normal_distribution<double> norm(0, 1);

        int n = loans.size();
        double total_ead = 0;
        for (auto& l : loans) total_ead += l.ead;

        // Pre-compute thresholds Phi^{-1}(PD_i)
        std::vector<double> thresh(n);
        for (int i = 0; i < n; ++i) thresh[i] = norm_inv(loans[i].pd);

        std::vector<double> losses(n_sim_);

        for (int s = 0; s < n_sim_; ++s) {
            double Z = norm(rng);   // systematic factor
            double L = 0;
            for (int i = 0; i < n; ++i) {
                double rho   = loans[i].rho_asset;
                double X_i   = std::sqrt(rho) * Z + std::sqrt(1.0 - rho) * norm(rng);
                if (X_i < thresh[i])
                    L += loans[i].ead * loans[i].lgd;
            }
            losses[s] = L / total_ead;  // as fraction of portfolio
        }

        std::sort(losses.begin(), losses.end());

        PortfolioResults r;
        r.loss_dist = losses;
        r.el  = std::accumulate(losses.begin(), losses.end(), 0.0) / n_sim_;
        double var_sum = 0;
        for (auto x : losses) var_sum += (x - r.el) * (x - r.el);
        r.ul = std::sqrt(var_sum / (n_sim_ - 1));

        int idx_95 = (int)(0.95 * n_sim_);
        int idx_99 = (int)(0.99 * n_sim_);
        r.var_95 = losses[idx_95];
        r.var_99 = losses[idx_99];

        // CVaR = mean of losses beyond VaR
        double s95 = 0; int c95 = 0;
        double s99 = 0; int c99 = 0;
        for (int i = idx_95; i < n_sim_; ++i) { s95 += losses[i]; ++c95; }
        for (int i = idx_99; i < n_sim_; ++i) { s99 += losses[i]; ++c99; }
        r.cvar_95 = c95 > 0 ? s95 / c95 : r.var_95;
        r.cvar_99 = c99 > 0 ? s99 / c99 : r.var_99;

        // Basel II capital requirement (sum of individual K * EAD)
        double cap = 0;
        for (auto& l : loans) cap += basel_capital(l.pd, l.lgd) * l.ead;
        r.capital_req = cap / total_ead;

        return r;
    }
};
