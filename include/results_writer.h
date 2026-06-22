#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include "classifiers.h"
#include "portfolio_mc.h"

inline void write_metrics_csv(const std::string& path,
                               const std::vector<std::string>& names,
                               const std::vector<ClassMetrics>& metrics) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    f << "model,accuracy,precision,recall,f1,auc,tp,fp,tn,fn\n";
    for (size_t i = 0; i < names.size(); ++i) {
        auto& m = metrics[i];
        f << names[i] << ","
          << m.accuracy << "," << m.precision << ","
          << m.recall   << "," << m.f1        << "," << m.auc << ","
          << m.tp << "," << m.fp << "," << m.tn << "," << m.fn << "\n";
    }
}

inline void write_proba_csv(const std::string& path,
                             const std::string& model_name,
                             const std::vector<double>& proba,
                             const std::vector<int>& y_true) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    f << "model,pred_proba,true_label\n";
    for (size_t i = 0; i < proba.size(); ++i)
        f << model_name << "," << proba[i] << "," << y_true[i] << "\n";
}

inline void write_all_probas_csv(const std::string& path,
                                  const std::vector<std::string>& names,
                                  const std::vector<std::vector<double>>& probas,
                                  const std::vector<int>& y_true) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    f << "true_label";
    for (auto& n : names) f << "," << n;
    f << "\n";
    for (size_t i = 0; i < y_true.size(); ++i) {
        f << y_true[i];
        for (size_t m = 0; m < names.size(); ++m) f << "," << probas[m][i];
        f << "\n";
    }
}

inline void write_feature_importance(const std::string& path,
    const std::vector<std::pair<std::string,double>>& imp) {
    std::ofstream f(path);
    f << "feature,importance\n";
    for (auto& [name, val] : imp)
        f << name << "," << std::fixed << std::setprecision(6) << val << "\n";
}

inline void write_loss_distribution(const std::string& path,
                                     const std::vector<double>& losses) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(8);
    f << "loss_pct\n";
    for (auto l : losses) f << l << "\n";
}

inline void write_portfolio_summary(const std::string& path,
                                     const PortfolioResults& r) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    f << "metric,value\n";
    f << "EL,"          << r.el          << "\n";
    f << "UL,"          << r.ul          << "\n";
    f << "VaR_95,"      << r.var_95      << "\n";
    f << "VaR_99,"      << r.var_99      << "\n";
    f << "CVaR_95,"     << r.cvar_95     << "\n";
    f << "CVaR_99,"     << r.cvar_99     << "\n";
    f << "Capital_Req," << r.capital_req << "\n";
}

inline void write_grade_pd(const std::string& path,
    const std::vector<std::string>& grades,
    const std::vector<double>& avg_pd,
    const std::vector<double>& emp_dr) {
    std::ofstream f(path);
    f << "grade,avg_pd,empirical_dr\n";
    for (size_t i = 0; i < grades.size(); ++i)
        f << grades[i] << "," << std::fixed << std::setprecision(6)
          << avg_pd[i] << "," << emp_dr[i] << "\n";
}
