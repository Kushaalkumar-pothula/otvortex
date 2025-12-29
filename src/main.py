import matplotlib.pyplot as plt
import numpy as np
import argparse
import engine
from matplotlib.animation import FFMpegWriter

"""
MHD Simulation with C++ Core Calculations
Orszag-Tang vortex problem

Build C++ module first:
    pip install pybind11
    python setup.py build_ext --inplace

Run simulation:
    python mhd_simulation.py --resolution 128 --end-time 0.5 --output simulation.mp4
"""


def getConserved(rho, vx, vy, P, Bx, By, gamma, vol):
    """Calculate conserved variables from primitive"""
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (
        (P - 0.5 * (Bx**2 + By**2)) / (gamma - 1)
        + 0.5 * rho * (vx**2 + vy**2)
        + 0.5 * (Bx**2 + By**2)
    ) * vol
    return Mass, Momx, Momy, Energy


def getPrimitive(Mass, Momx, Momy, Energy, Bx, By, gamma, vol):
    """Calculate primitive variables from conservative"""
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2) - 0.5 * (Bx**2 + By**2)) * (
        gamma - 1
    ) + 0.5 * (Bx**2 + By**2)
    return rho, vx, vy, P


def slopeLimit(f, dx, f_dx, f_dy):
    """Apply slope limiter"""
    R = -1
    L = 1
    
    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, L, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, (-(f - np.roll(f, R, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, L, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, (-(f - np.roll(f, R, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )
    return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """Extrapolate to face centers"""
    R = -1
    L = 1
    
    f_XL = f - f_dx * dx / 2
    f_XL = np.roll(f_XL, R, axis=0)
    f_XR = f + f_dx * dx / 2
    
    f_YL = f - f_dy * dx / 2
    f_YL = np.roll(f_YL, R, axis=1)
    f_YR = f + f_dy * dx / 2
    
    return f_XL, f_XR, f_YL, f_YR


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, Bx_L, Bx_R, By_L, By_R, gamma):
    """Calculate fluxes with Lax-Friedrichs"""
    en_L = (
        (P_L - 0.5 * (Bx_L**2 + By_L**2)) / (gamma - 1)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2)
        + 0.5 * (Bx_L**2 + By_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2)) / (gamma - 1)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2)
        + 0.5 * (Bx_R**2 + By_R**2)
    )
    
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    By_star = 0.5 * (By_L + By_R)
    
    P_star = (gamma - 1) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2)
    
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star - Bx_star * Bx_star
    flux_Momy = momx_star * momy_star / rho_star - Bx_star * By_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + By_star * momy_star
    ) / rho_star
    flux_By = (By_star * momx_star - Bx_star * momy_star) / rho_star
    
    c0_L = np.sqrt(gamma * (P_L - 0.5 * (Bx_L**2 + By_L**2)) / rho_L)
    c0_R = np.sqrt(gamma * (P_R - 0.5 * (Bx_R**2 + By_R**2)) / rho_R)
    ca_L = np.sqrt((Bx_L**2 + By_L**2) / rho_L)
    ca_R = np.sqrt((Bx_R**2 + By_R**2) / rho_R)
    cf_L = np.sqrt(0.5 * (c0_L**2 + ca_L**2) + 0.5 * np.sqrt((c0_L**2 + ca_L**2) ** 2))
    cf_R = np.sqrt(0.5 * (c0_R**2 + ca_R**2) + 0.5 * np.sqrt((c0_R**2 + ca_R**2) ** 2))
    C_L = cf_L + np.abs(vx_L)
    C_R = cf_R + np.abs(vx_R)
    C = np.maximum(C_L, C_R)
    
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    flux_By -= C * 0.5 * (By_L - By_R)
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_By


def main():
    """Main simulation function"""
    parser = argparse.ArgumentParser(description='MHD Orszag-Tang Vortex Simulation')
    parser.add_argument('--resolution', '-N', type=int, default=128,
                        help='Grid resolution (default: 128)')
    parser.add_argument('--end-time', '-t', type=float, default=0.5,
                        help='End time of simulation (default: 0.5)')
    parser.add_argument('--output', '-o', type=str, default='mhd_simulation.mp4',
                        help='Output video file (default: mhd_simulation.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Video DPI (default: 100)')
    parser.add_argument('--courant', '-c', type=float, default=0.4,
                        help='Courant factor (default: 0.4)')
    parser.add_argument('--no-slope-limit', action='store_true',
                        help='Disable slope limiting')
    parser.add_argument('--frames', type=int, default=300,
                        help='Target number of frames (default: 300)')
    
    args = parser.parse_args()
    
    # Simulation parameters
    N = args.resolution
    boxsize = 1.0
    gamma = 5 / 3
    courant_fac = args.courant
    t = 0
    tEnd = args.end_time
    # Calculate frame interval for desired number of frames
    tOut = tEnd / args.frames
    useSlopeLimiting = not args.no_slope_limit
    
    print(f"Starting Simulation")
    print(f"Resolution: {N}x{N}")
    print(f"End time: {tEnd}")
    print(f"Target frames: {args.frames}")
    print(f"Frame interval: {tOut:.6f}")
    print(f"Output: {args.output}")
    
    # Mesh
    dx = boxsize / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    Y, X = np.meshgrid(xlin, xlin)
    xlin_node = np.linspace(dx, boxsize, N)
    Yn, Xn = np.meshgrid(xlin_node, xlin_node)
    
    # Initial conditions
    rho = (gamma**2 / (4 * np.pi)) * np.ones(X.shape)
    vx = -np.sin(2 * np.pi * Y)
    vy = np.sin(2 * np.pi * X)
    P = (gamma / (4 * np.pi)) * np.ones(X.shape)
    
    # Magnetic field IC
    Az = np.cos(4 * np.pi * X) / (4 * np.pi * np.sqrt(4 * np.pi)) + np.cos(
        2 * np.pi * Y
    ) / (2 * np.pi * np.sqrt(4 * np.pi))
    
    # Use C++ function for curl
    bx, by = engine.getCurl(Az, dx)
    Bx, By = engine.getBavg(bx, by)
    
    P = P + 0.5 * (Bx**2 + By**2)
    
    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, Bx, By, gamma, vol)
    
    # Setup figure and video writer
    fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)
    writer = FFMpegWriter(fps=args.fps, metadata={'artist': 'Orszag-Tang Vortex'})
    
    # Pre-create image object for faster updates
    im = ax.imshow(rho.T, cmap='jet', origin='lower', vmin=0.06, vmax=0.5)
    cbar = plt.colorbar(im, ax=ax, label='Density')
    ax.set_aspect('equal')
    title = ax.set_title(f'Orszag-Tang Vortex - t = {t:.3f}')
    
    outputCount = 1
    frame_count = 0
    
    print("Running simulation...")
    
    # Main loop
    with writer.saving(fig, args.output, dpi=args.dpi):
        while t < tEnd:
            # Get primitive variables (using C++ for magnetic field averaging)
            Bx, By = engine.getBavg(bx, by)
            rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, Bx, By, gamma, vol)
            
            # Get time step
            c0 = np.sqrt(gamma * (P - 0.5 * (Bx**2 + By**2)) / rho)
            ca = np.sqrt((Bx**2 + By**2) / rho)
            cf = np.sqrt(0.5 * (c0**2 + ca**2) + 0.5 * np.sqrt((c0**2 + ca**2) ** 2))
            dt = courant_fac * np.min(dx / (cf + np.sqrt(vx**2 + vy**2)))
            
            plotThisTurn = False
            if t + dt > outputCount * tOut:
                dt = outputCount * tOut - t
                plotThisTurn = True
            
            # Calculate gradients (using C++ for faster computation)
            rho_dx, rho_dy = engine.getGradient(rho, dx)
            vx_dx, vx_dy = engine.getGradient(vx, dx)
            vy_dx, vy_dy = engine.getGradient(vy, dx)
            P_dx, P_dy = engine.getGradient(P, dx)
            Bx_dx, Bx_dy = engine.getGradient(Bx, dx)
            By_dx, By_dy = engine.getGradient(By, dx)
            
            # Slope limiting
            if useSlopeLimiting:
                rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
                vx_dx, vx_dy = slopeLimit(vx, dx, vx_dx, vx_dy)
                vy_dx, vy_dy = slopeLimit(vy, dx, vy_dx, vy_dy)
                P_dx, P_dy = slopeLimit(P, dx, P_dx, P_dy)
                Bx_dx, Bx_dy = slopeLimit(Bx, dx, Bx_dx, Bx_dy)
                By_dx, By_dy = slopeLimit(By, dx, By_dx, By_dy)
            
            # Time extrapolation
            rho_prime = rho - 0.5 * dt * (
                vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy
            )
            vx_prime = vx - 0.5 * dt * (
                vx * vx_dx + vy * vx_dy + (1 / rho) * P_dx
                - (2 * Bx / rho) * Bx_dx - (By / rho) * Bx_dy - (Bx / rho) * By_dy
            )
            vy_prime = vy - 0.5 * dt * (
                vx * vy_dx + vy * vy_dy + (1 / rho) * P_dy
                - (2 * By / rho) * By_dy - (Bx / rho) * By_dx - (By / rho) * Bx_dx
            )
            P_prime = P - 0.5 * dt * (
                (gamma * (P - 0.5 * (Bx**2 + By**2)) + By**2) * vx_dx
                - Bx * By * vy_dx + vx * P_dx
                + (gamma - 2) * (Bx * vx + By * vy) * Bx_dx
                - By * Bx * vx_dy
                + (gamma * (P - 0.5 * (Bx**2 + By**2)) + Bx**2) * vy_dy
                + vy * P_dy + (gamma - 2) * (Bx * vx + By * vy) * By_dy
            )
            Bx_prime = Bx - 0.5 * dt * (-By * vx_dy + Bx * vy_dy + vy * Bx_dy - vx * By_dy)
            By_prime = By - 0.5 * dt * (By * vx_dx - Bx * vy_dx - vy * Bx_dx + vx * By_dx)
            
            # Extrapolate to faces
            rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx)
            vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(vx_prime, vx_dx, vx_dy, dx)
            vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(vy_prime, vy_dx, vy_dy, dx)
            P_XL, P_XR, P_YL, P_YR = extrapolateInSpaceToFace(P_prime, P_dx, P_dy, dx)
            Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolateInSpaceToFace(Bx_prime, Bx_dx, Bx_dy, dx)
            By_XL, By_XR, By_YL, By_YR = extrapolateInSpaceToFace(By_prime, By_dx, By_dy, dx)
            
            # Compute fluxes
            flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_By_X = getFlux(
                rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR,
                Bx_XL, Bx_XR, By_XL, By_XR, gamma
            )
            flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_Bx_Y = getFlux(
                rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR,
                By_YL, By_YR, Bx_YL, Bx_YR, gamma
            )
            
            # Update solution (using C++ for flux application)
            Mass = engine.applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)[0]
            Momx = engine.applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)[0]
            Momy = engine.applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)[0]
            Energy = engine.applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)[0]
            bx, by = engine.constrainedTransport(bx, by, flux_By_X, flux_Bx_Y, dx, dt)
            
            t += dt
            
            # Save frame
            if plotThisTurn or t >= tEnd:
                divB = engine.getDiv(bx, by, dx)
                
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"Frame {frame_count}, t = {t:.4f}, mean |divB| = {np.mean(np.abs(divB)):.2e}")
                
                # Update plot efficiently
                im.set_data(rho.T)
                title.set_text(f'Orszag-Tang Vortex - t = {t:.3f}')
                
                writer.grab_frame()
                outputCount += 1
                frame_count += 1
    
    print(f"\nSimulation complete!")
    print(f"Total frames: {frame_count}")
    print(f"Video duration: {frame_count/args.fps:.2f} seconds")
    print(f"Video saved to {args.output}")
    plt.close()
    
    return 0


if __name__ == "__main__":
    main()