#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Helper function to get 2D array element
inline double get2D(const double* arr, int i, int j, int N) {
    return arr[i * N + j];
}

// Helper function for periodic boundary (roll)
inline int periodic(int i, int N) {
    return (i + N) % N;
}

py::tuple getCurl(py::array_t<double> Az, double dx) {
    auto Az_buf = Az.request();
    int N = Az_buf.shape[0];
    
    auto bx = py::array_t<double>({N, N});
    auto by = py::array_t<double>({N, N});
    
    double* Az_ptr = static_cast<double*>(Az_buf.ptr);
    double* bx_ptr = static_cast<double*>(bx.request().ptr);
    double* by_ptr = static_cast<double*>(by.request().ptr);
    
    // Python: bx = (Az - np.roll(Az, L, axis=1)) / dx where L=1
    // np.roll(Az, 1, axis=1) means: result[i,j] = Az[i, j-1]
    // So: bx[i,j] = (Az[i,j] - Az[i, j-1]) / dx
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int j_minus1 = periodic(j - 1, N);
            int i_minus1 = periodic(i - 1, N);
            
            bx_ptr[i * N + j] = (get2D(Az_ptr, i, j, N) - get2D(Az_ptr, i, j_minus1, N)) / dx;
            by_ptr[i * N + j] = -(get2D(Az_ptr, i, j, N) - get2D(Az_ptr, i_minus1, j, N)) / dx;
        }
    }
    
    return py::make_tuple(bx, by);
}

py::array_t<double> getDiv(py::array_t<double> bx, py::array_t<double> by, double dx) {
    auto bx_buf = bx.request();
    auto by_buf = by.request();
    int N = bx_buf.shape[0];
    
    auto divB = py::array_t<double>({N, N});
    
    double* bx_ptr = static_cast<double*>(bx_buf.ptr);
    double* by_ptr = static_cast<double*>(by_buf.ptr);
    double* divB_ptr = static_cast<double*>(divB.request().ptr);
    
    // Python: divB = (bx - np.roll(bx, L, axis=0) + by - np.roll(by, L, axis=1)) / dx
    // where L=1, so np.roll(bx, 1, axis=0) means: result[i,j] = bx[i-1,j]
    // Therefore: divB[i,j] = (bx[i,j] - bx[i-1,j] + by[i,j] - by[i,j-1]) / dx
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int i_minus1 = periodic(i - 1, N);
            int j_minus1 = periodic(j - 1, N);
            
            divB_ptr[i * N + j] = (get2D(bx_ptr, i, j, N) - get2D(bx_ptr, i_minus1, j, N) +
                                   get2D(by_ptr, i, j, N) - get2D(by_ptr, i, j_minus1, N)) / dx;
        }
    }
    
    return divB;
}

py::tuple getBavg(py::array_t<double> bx, py::array_t<double> by) {
    auto bx_buf = bx.request();
    auto by_buf = by.request();
    int N = bx_buf.shape[0];
    
    auto Bx = py::array_t<double>({N, N});
    auto By = py::array_t<double>({N, N});
    
    double* bx_ptr = static_cast<double*>(bx_buf.ptr);
    double* by_ptr = static_cast<double*>(by_buf.ptr);
    double* Bx_ptr = static_cast<double*>(Bx.request().ptr);
    double* By_ptr = static_cast<double*>(By.request().ptr);
    
    // Python: Bx = 0.5 * (bx + np.roll(bx, L, axis=0)) where L=1
    // np.roll(bx, 1, axis=0) means: result[i,j] = bx[i-1,j]
    // So: Bx[i,j] = 0.5 * (bx[i,j] + bx[i-1,j])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int i_minus1 = periodic(i - 1, N);
            int j_minus1 = periodic(j - 1, N);
            
            Bx_ptr[i * N + j] = 0.5 * (get2D(bx_ptr, i, j, N) + get2D(bx_ptr, i_minus1, j, N));
            By_ptr[i * N + j] = 0.5 * (get2D(by_ptr, i, j, N) + get2D(by_ptr, i, j_minus1, N));
        }
    }
    
    return py::make_tuple(Bx, By);
}

py::tuple getGradient(py::array_t<double> f, double dx) {
    auto f_buf = f.request();
    int N = f_buf.shape[0];
    
    auto f_dx = py::array_t<double>({N, N});
    auto f_dy = py::array_t<double>({N, N});
    
    double* f_ptr = static_cast<double*>(f_buf.ptr);
    double* f_dx_ptr = static_cast<double*>(f_dx.request().ptr);
    double* f_dy_ptr = static_cast<double*>(f_dy.request().ptr);
    
    // Python: f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2*dx)
    // where R=-1, L=1
    // np.roll(f, -1, axis=0) means: result[i,j] = f[i+1,j]
    // np.roll(f, 1, axis=0) means: result[i,j] = f[i-1,j]
    // So: f_dx[i,j] = (f[i+1,j] - f[i-1,j]) / (2*dx)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int i_plus1 = periodic(i + 1, N);
            int i_minus1 = periodic(i - 1, N);
            int j_plus1 = periodic(j + 1, N);
            int j_minus1 = periodic(j - 1, N);
            
            f_dx_ptr[i * N + j] = (get2D(f_ptr, i_plus1, j, N) - get2D(f_ptr, i_minus1, j, N)) / (2.0 * dx);
            f_dy_ptr[i * N + j] = (get2D(f_ptr, i, j_plus1, N) - get2D(f_ptr, i, j_minus1, N)) / (2.0 * dx);
        }
    }
    
    return py::make_tuple(f_dx, f_dy);
}

py::tuple applyFluxes(py::array_t<double> F, py::array_t<double> flux_F_X, 
                      py::array_t<double> flux_F_Y, double dx, double dt) {
    auto F_buf = F.request();
    auto flux_X_buf = flux_F_X.request();
    auto flux_Y_buf = flux_F_Y.request();
    int N = F_buf.shape[0];
    
    auto F_out = py::array_t<double>({N, N});
    
    double* F_ptr = static_cast<double*>(F_buf.ptr);
    double* flux_X_ptr = static_cast<double*>(flux_X_buf.ptr);
    double* flux_Y_ptr = static_cast<double*>(flux_Y_buf.ptr);
    double* F_out_ptr = static_cast<double*>(F_out.request().ptr);
    
    // Copy input to output
    for (int i = 0; i < N * N; i++) {
        F_out_ptr[i] = F_ptr[i];
    }
    
    // Python code:
    // F += -dt * dx * flux_F_X
    // F += dt * dx * np.roll(flux_F_X, L, axis=0)  where L=1
    // np.roll(flux_F_X, 1, axis=0) means: result[i,j] = flux_F_X[i-1,j]
    // So: F[i,j] += dt * dx * flux_F_X[i-1,j]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int i_minus1 = periodic(i - 1, N);
            int j_minus1 = periodic(j - 1, N);
            
            F_out_ptr[i * N + j] += -dt * dx * get2D(flux_X_ptr, i, j, N);
            F_out_ptr[i * N + j] += dt * dx * get2D(flux_X_ptr, i_minus1, j, N);
            F_out_ptr[i * N + j] += -dt * dx * get2D(flux_Y_ptr, i, j, N);
            F_out_ptr[i * N + j] += dt * dx * get2D(flux_Y_ptr, i, j_minus1, N);
        }
    }
    
    return py::make_tuple(F_out);
}

py::tuple constrainedTransport(py::array_t<double> bx, py::array_t<double> by,
                               py::array_t<double> flux_By_X, py::array_t<double> flux_Bx_Y,
                               double dx, double dt) {
    auto bx_buf = bx.request();
    auto by_buf = by.request();
    auto flux_By_X_buf = flux_By_X.request();
    auto flux_Bx_Y_buf = flux_Bx_Y.request();
    int N = bx_buf.shape[0];
    
    auto bx_out = py::array_t<double>({N, N});
    auto by_out = py::array_t<double>({N, N});
    auto Ez = py::array_t<double>({N, N});
    
    double* bx_ptr = static_cast<double*>(bx_buf.ptr);
    double* by_ptr = static_cast<double*>(by_buf.ptr);
    double* flux_By_X_ptr = static_cast<double*>(flux_By_X_buf.ptr);
    double* flux_Bx_Y_ptr = static_cast<double*>(flux_Bx_Y_buf.ptr);
    double* bx_out_ptr = static_cast<double*>(bx_out.request().ptr);
    double* by_out_ptr = static_cast<double*>(by_out.request().ptr);
    double* Ez_ptr = static_cast<double*>(Ez.request().ptr);
    
    // Python: Ez = 0.25 * (-flux_By_X - np.roll(flux_By_X, R, axis=1) 
    //                      + flux_Bx_Y + np.roll(flux_Bx_Y, R, axis=0))
    // where R=-1
    // np.roll(flux_By_X, -1, axis=1) means: result[i,j] = flux_By_X[i, j+1]
    // np.roll(flux_Bx_Y, -1, axis=0) means: result[i,j] = flux_Bx_Y[i+1, j]
    // So: Ez[i,j] = 0.25 * (-flux_By_X[i,j] - flux_By_X[i,j+1]
    //                       + flux_Bx_Y[i,j] + flux_Bx_Y[i+1,j])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int i_plus1 = periodic(i + 1, N);
            int j_plus1 = periodic(j + 1, N);
            
            Ez_ptr[i * N + j] = 0.25 * (
                -get2D(flux_By_X_ptr, i, j, N) - get2D(flux_By_X_ptr, i, j_plus1, N) +
                get2D(flux_Bx_Y_ptr, i, j, N) + get2D(flux_Bx_Y_ptr, i_plus1, j, N)
            );
        }
    }
    
    // Get curl of -Ez to update magnetic field
    // Python: dbx, dby = getCurl(-Ez, dx)
    // Then: bx += dt * dbx; by += dt * dby
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int j_minus1 = periodic(j - 1, N);
            int i_minus1 = periodic(i - 1, N);
            
            // curl of -Ez:
            // dbx = ((-Ez)[i,j] - (-Ez)[i,j-1]) / dx = -(Ez[i,j] - Ez[i,j-1]) / dx
            // dby = -((-Ez)[i,j] - (-Ez)[i-1,j]) / dx = (Ez[i,j] - Ez[i-1,j]) / dx
            double dbx = -(get2D(Ez_ptr, i, j, N) - get2D(Ez_ptr, i, j_minus1, N)) / dx;
            double dby = (get2D(Ez_ptr, i, j, N) - get2D(Ez_ptr, i_minus1, j, N)) / dx;
            
            bx_out_ptr[i * N + j] = get2D(bx_ptr, i, j, N) + dt * dbx;
            by_out_ptr[i * N + j] = get2D(by_ptr, i, j, N) + dt * dby;
        }
    }
    
    return py::make_tuple(bx_out, by_out);
}

PYBIND11_MODULE(engine, m) {
    m.doc() = "MHD core computation functions in C++";
    
    m.def("getCurl", &getCurl, "Calculate discrete curl",
          py::arg("Az"), py::arg("dx"));
    
    m.def("getDiv", &getDiv, "Calculate discrete divergence",
          py::arg("bx"), py::arg("by"), py::arg("dx"));
    
    m.def("getBavg", &getBavg, "Calculate volume-averaged magnetic field",
          py::arg("bx"), py::arg("by"));
    
    m.def("getGradient", &getGradient, "Calculate gradients",
          py::arg("f"), py::arg("dx"));
    
    m.def("applyFluxes", &applyFluxes, "Apply fluxes to conserved variables",
          py::arg("F"), py::arg("flux_F_X"), py::arg("flux_F_Y"), 
          py::arg("dx"), py::arg("dt"));
    
    m.def("constrainedTransport", &constrainedTransport, 
          "Apply constrained transport to magnetic fields",
          py::arg("bx"), py::arg("by"), py::arg("flux_By_X"), 
          py::arg("flux_Bx_Y"), py::arg("dx"), py::arg("dt"));
}