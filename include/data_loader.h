#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "matrix.h"

// ─────────────────────────────────────────────────────────────────────────────
// Loan record matching the Lending Club CSV schema
// ─────────────────────────────────────────────────────────────────────────────
struct LoanRecord {
    int    id;
    double loan_amnt;
    int    term;          // 36 or 60
    double int_rate;
    int    grade;         // 0=A .. 6=G
    int    emp_length;
    int    home_own;      // 0=MORTGAGE 1=OWN 2=RENT
    double annual_inc;
    int    purpose;       // encoded
    double dti;
    int    delinq_2yrs;
    double fico;          // midpoint of low/high
    int    inq_last_6m;
    int    open_acc;
    int    pub_rec;
    double revol_util;
    int    label;         // 1=default, 0=paid
};

inline int encode_grade(const std::string& g) {
    if (g=="A") return 0; if (g=="B") return 1; if (g=="C") return 2;
    if (g=="D") return 3; if (g=="E") return 4; if (g=="F") return 5;
    return 6;
}
inline int encode_home(const std::string& h) {
    if (h=="MORTGAGE") return 0; if (h=="OWN") return 1; return 2;
}
inline int encode_purpose(const std::string& p) {
    static const std::vector<std::string> ps = {
        "debt_consolidation","credit_card","home_improvement",
        "major_purchase","medical","small_business","vacation","other"
    };
    for (int i = 0; i < (int)ps.size(); ++i) if (ps[i] == p) return i;
    return 7;
}

// Parse term string "36 months" -> 36
inline int parse_term(const std::string& t) {
    return std::stoi(t.substr(0, t.find(' ')));
}

inline std::vector<LoanRecord> load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(f, line); // header

    std::vector<LoanRecord> records;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> fields;
        while (std::getline(ss, tok, ',')) fields.push_back(tok);
        if (fields.size() < 19) continue;

        LoanRecord r;
        try {
            r.id          = std::stoi(fields[0]);
            r.loan_amnt   = std::stod(fields[1]);
            r.term        = parse_term(fields[2]);
            r.int_rate    = std::stod(fields[3]);
            r.grade       = encode_grade(fields[4]);
            r.emp_length  = std::stoi(fields[5]);
            r.home_own    = encode_home(fields[6]);
            r.annual_inc  = std::stod(fields[7]);
            r.purpose     = encode_purpose(fields[8]);
            r.dti         = std::stod(fields[9]);
            r.delinq_2yrs = std::stoi(fields[10]);
            r.fico        = (std::stod(fields[11]) + std::stod(fields[12])) / 2.0;
            r.inq_last_6m = std::stoi(fields[13]);
            r.open_acc    = std::stoi(fields[14]);
            r.pub_rec     = std::stoi(fields[15]);
            r.revol_util  = std::stod(fields[16]);
            // fields[17] = loan_status (string)
            r.label       = std::stoi(fields[18]);
        } catch (...) { continue; }

        records.push_back(r);
    }
    return records;
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature matrix builder — numeric features only
// ─────────────────────────────────────────────────────────────────────────────
inline const std::vector<std::string> FEATURE_NAMES = {
    "loan_amnt", "int_rate", "grade", "emp_length", "home_own",
    "annual_inc", "dti", "delinq_2yrs", "fico", "inq_last_6m",
    "open_acc", "pub_rec", "revol_util", "term"
};

// Convert records to (X, y) — standardizes features (zero mean, unit variance)
inline void build_features(const std::vector<LoanRecord>& recs,
                            Matrix& X, std::vector<int>& y,
                            std::vector<double>& mu_out,
                            std::vector<double>& sigma_out,
                            bool standardize = true) {
    int n = (int)recs.size();
    int p = (int)FEATURE_NAMES.size();
    X = Matrix(n, p);
    y.resize(n);

    auto fill_row = [&](int i, const LoanRecord& r) {
        X.at(i, 0)  = r.loan_amnt;
        X.at(i, 1)  = r.int_rate;
        X.at(i, 2)  = r.grade;
        X.at(i, 3)  = r.emp_length;
        X.at(i, 4)  = r.home_own;
        X.at(i, 5)  = r.annual_inc;
        X.at(i, 6)  = r.dti;
        X.at(i, 7)  = r.delinq_2yrs;
        X.at(i, 8)  = r.fico;
        X.at(i, 9)  = r.inq_last_6m;
        X.at(i, 10) = r.open_acc;
        X.at(i, 11) = r.pub_rec;
        X.at(i, 12) = r.revol_util;
        X.at(i, 13) = r.term;
        y[i]        = r.label;
    };

    for (int i = 0; i < n; ++i) fill_row(i, recs[i]);

    // Standardize per column
    mu_out.resize(p); sigma_out.resize(p);
    if (standardize) {
        for (int j = 0; j < p; ++j) {
            auto col = X.col(j);
            double m = vec::mean(col);
            double s = std::sqrt(vec::variance(col));
            if (s < 1e-10) s = 1.0;
            mu_out[j] = m; sigma_out[j] = s;
            for (int i = 0; i < n; ++i) X.at(i,j) = (X.at(i,j) - m) / s;
        }
    }
}

// Apply stored standardization to new data
inline void apply_standardize(Matrix& X,
                               const std::vector<double>& mu,
                               const std::vector<double>& sigma) {
    for (int j = 0; j < X.cols; ++j)
        for (int i = 0; i < X.rows; ++i)
            X.at(i,j) = (X.at(i,j) - mu[j]) / sigma[j];
}

// Train/test split
inline void train_test_split(const Matrix& X, const std::vector<int>& y,
                              Matrix& X_train, std::vector<int>& y_train,
                              Matrix& X_test,  std::vector<int>& y_test,
                              double test_frac = 0.20, unsigned seed = 42) {
    int n = X.rows;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    int n_test  = (int)(n * test_frac);
    int n_train = n - n_test;

    X_train = Matrix(n_train, X.cols);
    X_test  = Matrix(n_test,  X.cols);
    y_train.resize(n_train); y_test.resize(n_test);

    for (int i = 0; i < n_train; ++i) {
        int r = idx[i];
        for (int j = 0; j < X.cols; ++j) X_train.at(i,j) = X.at(r,j);
        y_train[i] = y[r];
    }
    for (int i = 0; i < n_test; ++i) {
        int r = idx[n_train + i];
        for (int j = 0; j < X.cols; ++j) X_test.at(i,j) = X.at(r,j);
        y_test[i] = y[r];
    }
}
