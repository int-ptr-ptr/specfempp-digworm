# Simple Error Experiment

This is a minor grid of simulations to compute how much efficiency can we pull with a nonconforming fluid-solid interface compared to the base conforming simulation.

## Setup

We use

- A rectangular mesh with a solid lower half and fluid upper half
- Non-absorbing boundaries
- Nondimensionalization:
  - A point source of frequency `f0`
  - A base topographical wavelength of `L0`
- topography is single sinusoid, harmonics of `L0`. amplitude of `a0`, fixed factor of `L0` (perhaps L0/(2pi * k) for some k).
- height is fixed, some multiple of `L0` to keep `a0/L0` fixed and small enough for aspect ratio to not change too much along bathy.
- width is fixed, some large enough multiple of `L0` so boundaries don't play too big of a role in the simulation
- vary source location (both in fluid and solid)
- fixed receivers.
- solid p-wave speed fixed to `L0 * f0`. Set s-wave to reasonable value (should we let this vary?).
- fluid p-wave speed varies.

We want to see if there is a specific guideline to choosing mesh grid size between media, as well as time step size.
