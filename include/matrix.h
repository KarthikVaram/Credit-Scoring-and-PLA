#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
// Dense matrix (row-major) with basic linear algebra
// ─────────────────────────────────────────────────────────────────────────────
class Matrix {
public:
    int rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    double&       at(int r, int c)       { return data[r * cols + c]; }
    const double& at(int r, int c) const { return data[r * cols + c]; }

    // Column vector access
    std::vector<double> col(int c) const {
        std::vector<double> v(rows);
        for (int r = 0; r < rows; ++r) v[r] = at(r, c);
        return v;
    }

    // Row vector access
    std::vector<double> row(int r) const {
        return std::vector<double>(data.begin() + r*cols,
                                   data.begin() + r*cols + cols);
    }

    // Matrix transpose
    Matrix T() const {
        Matrix out(cols, rows);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.at(c, r) = at(r, c);
        return out;
    }

    // Matrix multiply
    Matrix operator*(const Matrix& B) const {
        if (cols != B.rows)
            throw std::runtime_error("Matrix dimension mismatch in multiply");
        Matrix out(rows, B.cols, 0.0);
        for (int i = 0; i < rows; ++i)
            for (int k = 0; k < cols; ++k)
                for (int j = 0; j < B.cols; ++j)
                    out.at(i, j) += at(i, k) * B.at(k, j);
        return out;
    }

    // Matrix-vector multiply
    std::vector<double> operator*(const std::vector<double>& v) const {
        if ((int)v.size() != cols)
            throw std::runtime_error("Matrix-vector dimension mismatch");
        std::vector<double> out(rows, 0.0);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                out[i] += at(i, j) * v[j];
        return out;
    }

    Matrix operator+(const Matrix& B) const {
        Matrix out(rows, cols);
        for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] + B.data[i];
        return out;
    }
    Matrix operator-(const Matrix& B) const {
        Matrix out(rows, cols);
        for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] - B.data[i];
        return out;
    }
    Matrix operator*(double s) const {
        Matrix out(rows, cols);
        for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] * s;
        return out;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Vector utilities
// ─────────────────────────────────────────────────────────────────────────────
namespace vec {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm(const std::vector<double>& a) {
    return std::sqrt(dot(a, a));
}

inline double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double variance(const std::vector<double>& v) {
    double m = mean(v);
    double s = 0;
    for (auto x : v) s += (x - m) * (x - m);
    return s / (v.size() - 1);
}

inline std::vector<double> add(const std::vector<double>& a,
                                const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

inline std::vector<double> sub(const std::vector<double>& a,
                                const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
    return out;
}

inline std::vector<double> scale(const std::vector<double>& a, double s) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * s;
    return out;
}

} // namespace vec

// ─────────────────────────────────────────────────────────────────────────────
// LU Decomposition (for matrix inversion)
// ─────────────────────────────────────────────────────────────────────────────
inline bool lu_decompose(Matrix A, std::vector<int>& piv) {
    int n = A.rows;
    piv.resize(n);
    for (int i = 0; i < n; ++i) piv[i] = i;

    for (int k = 0; k < n; ++k) {
        // Partial pivoting
        int maxRow = k;
        double maxVal = std::abs(A.at(k, k));
        for (int i = k+1; i < n; ++i)
            if (std::abs(A.at(i, k)) > maxVal) { maxVal = std::abs(A.at(i, k)); maxRow = i; }

        if (maxVal < 1e-12) return false; // singular

        if (maxRow != k) {
            for (int j = 0; j < n; ++j) std::swap(A.at(k, j), A.at(maxRow, j));
            std::swap(piv[k], piv[maxRow]);
        }

        for (int i = k+1; i < n; ++i) {
            A.at(i, k) /= A.at(k, k);
            for (int j = k+1; j < n; ++j)
                A.at(i, j) -= A.at(i, k) * A.at(k, j);
        }
    }
    return true;
}

// Solve Ax = b given LU-decomposed A (Gauss elimination, returns false if singular)
inline Matrix invert(Matrix A) {
    int n = A.rows;
    Matrix inv(n, n, 0.0);
    for (int i = 0; i < n; ++i) inv.at(i, i) = 1.0;

    // Augmented Gaussian elimination with full pivoting
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col+1; row < n; ++row)
            if (std::abs(A.at(row, col)) > std::abs(A.at(pivot, col)))
                pivot = row;
        for (int j = 0; j < n; ++j) {
            std::swap(A.at(col, j),   A.at(pivot, j));
            std::swap(inv.at(col, j), inv.at(pivot, j));
        }
        double d = A.at(col, col);
        if (std::abs(d) < 1e-12) throw std::runtime_error("Matrix is singular");
        for (int j = 0; j < n; ++j) { A.at(col,j) /= d; inv.at(col,j) /= d; }
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = A.at(row, col);
            for (int j = 0; j < n; ++j) {
                A.at(row,j)   -= f * A.at(col,j);
                inv.at(row,j) -= f * inv.at(col,j);
            }
        }
    }
    return inv;
}
