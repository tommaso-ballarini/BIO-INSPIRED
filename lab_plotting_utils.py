# ======================================================================
#  file: lab_plotting_utils.py
# Contiene tutto il codice della tua "Cella 2" (i plotter)
# ======================================================================

from pylab import *
import sys
import numpy as np
import inspyred.ec.analysis

# Nota: plot_1D e plot_2D non verranno usati perché il nostro
# problema ha 20 dimensioni, ma li includiamo per coerenza.

def plot_1D(axis, problem, x_limits) :
    dx = (x_limits[1] - x_limits[0])/200.0
    x = arange(x_limits[0], x_limits[1]+dx, dx)
    x = x.reshape(len(x),1)
    y = problem.evaluator(x, None)
    axis.plot(x,y,'-b')

def plot_2D(axis, problem, x_limits) :
    dx = (x_limits[1] - x_limits[0])/50.0
    x = arange(x_limits[0], x_limits[1]+dx, dx)
    z = asarray( [problem.evaluator([[i,j] for i in x], None) for j in x])
    return axis.contourf(x, x, z, 64, cmap=cm.hot_r)

def plot_results_1D(problem, individuals_1, fitnesses_1,
                    individuals_2, fitnesses_2, title_1, title_2, args) :
    fig = figure(args["fig_title"] + ' (initial and final population)')
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(individuals_1, fitnesses_1, '.b', markersize=7)
    lim = max(np.array(list(map(abs,ax1.get_xlim()))))

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(individuals_2, fitnesses_2, '.b', markersize=7)
    lim = max([lim] + np.array(list(map(abs, ax2.get_xlim()))))

    ax1.set_xlim(-lim, lim)
    ax2.set_xlim(-lim, lim)

    plot_1D(ax1, problem, [-lim, lim])
    plot_1D(ax2, problem, [-lim, lim])

    ax1.set_ylabel('Fitness')
    ax2.set_ylabel('Fitness')
    ax1.set_title(title_1)
    ax2.set_title(title_2)
    fig.tight_layout()

def plot_results_2D(problem, individuals_1, individuals_2,
                    title_1, title_2, args) :
    fig = figure(args["fig_title"] + ' (initial and final population)')
    ax1 = fig.add_subplot(2,1,1, aspect='equal')
    ax1.plot(individuals_1[:,0], individuals_1[:,1], '.b', markersize=7)
    lim = max(np.array(list(map(abs,ax1.get_xlim()))) + np.array(list(map(abs,ax1.get_ylim()))))

    ax2 = fig.add_subplot(2,1,2, aspect='equal')
    ax2.plot(individuals_2[:,0], individuals_2[:,1], '.b', markersize=7)
    lim = max([lim] +
              np.array(list(map(abs,ax2.get_xlim()))) +
              np.array(list(map(abs,ax2.get_ylim()))))

    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_title(title_1)
    ax1.locator_params(nbins=5)

    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_title(title_2)
    ax2.set_xlabel('x0')
    ax2.set_ylabel('x1')
    ax2.locator_params(nbins=5)

    plot_2D(ax1, problem, [-lim, lim])
    c = plot_2D(ax2, problem, [-lim, lim])
    fig.subplots_adjust(right=0.8)
    fig.tight_layout()
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    colorbar_ = colorbar(c, cax=cax)
    colorbar_.ax.set_ylabel('Fitness')


# Questo è il 'plot_observer' della tua Cella 2
def plot_observer(population, num_generations, num_evaluations, args):
    import matplotlib.pyplot as plt
    import numpy

    stats = inspyred.ec.analysis.fitness_statistics(population)
    best_fitness = stats['best']
    worst_fitness = stats['worst']
    median_fitness = stats['median']
    average_fitness = stats['mean']
    colors = ['black', 'blue', 'green', 'red']
    labels = ['average', 'median', 'best', 'worst']
    data = []
    if num_generations == 0:
        figure(args["fig_title"] + ' (fitness trend)')
        plt.ion()
        data = [[num_evaluations], [average_fitness], [median_fitness], [best_fitness], [worst_fitness]]
        lines = []
        for i in range(4):
            line, = plt.plot(data[0], data[i+1], color=colors[i], label=labels[i])
            lines.append(line)
        args['plot_data'] = data
        args['plot_lines'] = lines
        plt.xlabel('Evaluations')
        plt.ylabel('Fitness')
    else:
        data = args['plot_data']
        data[0].append(num_evaluations)
        data[1].append(average_fitness)
        data[2].append(median_fitness)
        data[3].append(best_fitness)
        data[4].append(worst_fitness)
        lines = args['plot_lines']
        for i, line in enumerate(lines):
            line.set_xdata(numpy.array(data[0]))
            line.set_ydata(numpy.array(data[i+1]))
        args['plot_data'] = data
        args['plot_lines'] = lines
    ymin = min([min(d) for d in data[1:]])
    ymax = max([max(d) for d in data[1:]])
    yrange = ymax - ymin
    # Gestisce il caso in cui yrange è 0
    if yrange == 0:
        yrange = 1
        
    plt.xlim((0, max(1, num_evaluations))) # Evita xlim(0,0)
    plt.ylim((ymin - 0.1*yrange, ymax + 0.1*yrange))
    plt.draw()
    plt.legend()