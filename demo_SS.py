import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapz
from scipy.stats import norm
from scipy.misc import logsumexp


def slice_sample(x_start, pdf_target, D, num_samples=1, burn=1, lag=1,
                 w=1.0, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)

    M = {'u': [], 'r': [], 'a_out': [], 'b_out': [], 'x_proposal': [], 'samples': []}
    x = x_start

    num_iters = 0
    while len(M['samples']) < num_samples:
        num_iters += 1
        u = rng.rand() * pdf_target(x)
        r = rng.rand()
        a, b, r, a_out, b_out = _find_slice_interval(
            pdf_target, x, u, D, r, w=w)

        x_proposal = []

        while True:
            x_prime = rng.uniform(a, b)
            x_proposal.append(x)
            if pdf_target(x_prime) > u:
                x = x_prime
                break
            else:
                if x_prime > x:
                    b = x_prime
                else:
                    a = x_prime

        if burn <= num_iters and num_iters % lag == 0:
            M['u'].append(u)
            M['r'].append(r)
            M['a_out'].append(a_out)
            M['b_out'].append(b_out)
            M['x_proposal'].append(x_proposal)
            M['samples'].append(x)

    return M['samples'], M


def _find_slice_interval(f, x, u, D, r, w=1.0):
    a = x - r * w
    b = x + (1 - r) * w
    a_out = [a]
    b_out = [b]
    if a < D[0]:
        a = D[0]
        a_out[-1] = a
    else:
        while f(a) > u:
            a -= w
            a_out.append(a)
            if a < D[0]:
                a = D[0]
                a_out[-1] = a
                break
    if b > D[1]:
        b = D[1]
        b_out[-1] = b
    else:
        while f(b) > u:
            b += w
            b_out.append(b)
            if b > D[1]:
                b = D[1]
                b_out[-1] = b
                break
    return a, b, r, a_out, b_out


def dist_x():
    pdf = lambda x: np.sum([
        .2 * norm.pdf(x, 1, .5),
        .5 * norm.pdf(x, 4, .75),
        .3 * norm.pdf(x, 7, .9)])
    domain = (0, float('inf'))
    return domain, pdf


D, pdf_target = dist_x()
x_start = 4
samples, metadata = slice_sample(
    x_start, pdf_target, D, num_samples=100, burn=1, lag=1, w=.5,
    rng=np.random.RandomState(5))
fig, ax = plt.subplots()
ax.grid()
xvals = np.linspace(0.1, 10, 100)
yvals = list(map(pdf_target, xvals))
ax.plot(xvals, yvals / trapz(yvals, xvals), lw=3, alpha=.8, c='m',
        label='Target Density')
ax.set_xlim([0, 10])
ax.set_ylim([0, ax.get_ylim()[1]])

ax.legend(loc='upper left', framealpha=0)

plt.ion()
plt.show()

ptime = .5

last_x = x_start
for i in range(len(samples)):
    u = metadata['u'][i]
    r = metadata['r'][i]
    a_out = metadata['a_out'][i]
    b_out = metadata['b_out'][i]
    x_proposal = metadata['x_proposal'][i]
    sample = samples[i]
    to_delete = []
    plt.pause(ptime)
    to_delete.append(
        ax.vlines(last_x, 0, pdf_target(last_x) / trapz(yvals, xvals),
                  color='navy', linewidth=1.5))
    plt.pause(ptime)
    to_delete.append(
        ax.scatter(last_x, u, color='navy'))
    for a in a_out:
        plt.pause(ptime / 2.)
        to_delete.append(ax.hlines(u, a, last_x))
        to_delete.append(ax.vlines(a, u, u + .005))
    for b in b_out:
        plt.pause(ptime / 2.)
        to_delete.append(ax.hlines(u, last_x, b))
        to_delete.append(ax.vlines(b, u, u + .005))
    plt.pause(ptime)
    ax.scatter(sample, u, color='navy', alpha=0.5)
    plt.pause(ptime)
    ax.vlines(sample, 0, 0.005, linewidth=2, color='m')
    last_x = sample
    plt.pause(ptime)
    for td in to_delete:
        td.remove()