# pip install python-ternary is necessary
# Global font settings
font = {'weight': 'normal',
        'size': 16}
plt.rc('font', **font)
plt.rcParams['figure.dpi'] = 300

# Create figure and ternary axis
fig, ax = plt.subplots(figsize=(3.93701, 3.1496), dpi=1200)
figure, tax = ternary.figure(ax=ax, scale=1)
figure.set_size_inches(5, 5)

# Boundary and grid
tax.boundary(linewidth=1.0)
tax.gridlines(color="black", multiple=0.1, linewidth=0.5)

# Initial ticks
tax.ticks(axis='lbr',
          multiple=0.2,
          linewidth=1,
          tick_formats="%.1f",
          fontsize=14,
          offset=0.03)   # move labels further out



# Example PC-SAFT plot (replace x, w, z with your data arrays)
n = len(x)
X, W, Z = np.zeros([n, 3]), np.zeros([n, 3]), np.zeros([n, 3])
for i in range(n):
    X[i, :] = x[i][0], x[i][1], x[i][2]
    W[i, :] = w[i][0], w[i][1], w[i][2]
    Z[i, :] = z[i][0], z[i][1], z[i][2]

tax.plot(X, color="#FC7725", linewidth=2.0)
tax.plot(W, color="#FC7725", linewidth=2.0)
#tax.plot(Z, color="k", linewidth=2.0)

# PC-SAFT tie-line
nT = len(x)
nT0 = int(nT/1.)
print(f'Tie Lines System: {saft.pure_eos[0].pure.name} (1) + {saft.pure_eos[1].pure.name} (2) + {saft.pure_eos[2].pure.name} (3) at T = {T-273.15:.2f} °C and P = {P/1e5:.3f} bar')
print("--------------------------------------------------------------")
for i in range(1, nT0, 10):
    p1 = (x[i][0], x[i][1], x[i][2])
    p2 = (w[i][0], w[i][1], w[i][2])
    tax.line(p1, p2, linewidth = 1.5, markersize = 8, 
                markeredgewidth = 1, markeredgecolor = "black", 
                marker = "o", color = "#49C60A", linestyle = ":")
    # Print tie-line compositions and p1 and p2
    print(f'Tie-line {i}: x = [{p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}], w = [{p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f}]')
 
# Read experimental data
x1, x2, x3, w1, w2, w3 = np.loadtxt(f'exp_data/24_{saft.pure_eos[0].pure.name}+{saft.pure_eos[1].pure.name}+{saft.pure_eos[2].pure.name}.dat', unpack=True, comments=';')
n = len(x1)
X_exp, W_exp = np.zeros([n, 3]), np.zeros([n, 3])
for i in range(n):
    p1 = (x1[i], x2[i], x3[i])
    p2 = (w1[i], w2[i], w3[i])
    tax.line(p1, p2, linewidth = 1.5, markersize = 8, 
                markeredgewidth = 1, markeredgecolor = "black", 
                marker = "s", color = "#0000FF", linestyle = "--")

# Axis labels moved further away
tax.left_axis_label(f'{saft.pure_eos[2].pure.name} (3)', fontsize=16, offset=0.20)
tax.right_axis_label(f'{saft.pure_eos[1].pure.name} (2)', fontsize=16, offset=0.20)
tax.bottom_axis_label(f'{saft.pure_eos[0].pure.name} (1)',
                      position=(0.55, -0.1, 0.5),  # custom barycentric coords
                      fontsize=16)




# Final display
plt.axis("off")

plt.savefig(f'plots/LLE_{saft.pure_eos[0].pure.name}+{saft.pure_eos[1].pure.name}+{saft.pure_eos[2].pure.name}_T{T-273.15:.0f}C_P{P/1e5:.3f}bar.pdf', bbox_inches='tight', dpi=1200)
plt.show()