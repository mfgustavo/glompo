

def test_exp_kernel():
    """ Tests the exp kernel. """

    import matplotlib.pyplot as plt
    import numpy as np
    from core.expkernel import ExpKernel
    from testsworkdir.toyopti import ToyOptimizer
    from core.gpr import GaussianProcessRegression


    fig0, axlst0 = plt.subplots(4, 2, figsize=(20, 20))
    fig1, axlst1 = plt.subplots(4, 2, figsize=(20, 20))
    axlst = (axlst0, axlst1)

    kernel = ExpKernel(alpha=0.5,
                       beta=0.5)
    engine = ToyOptimizer(init=None,
                          min_=None,
                          steps_to_conv=None,
                          noise=0.15,
                          elitism=True)
    gpr = GaussianProcessRegression(kernel=kernel,
                                    dims=1,
                                    sigma_noise=0.0,
                                    mean=None,
                                    cache_results=True)

    ntrain = 8

    t = np.arange(0, 500)
    y = np.array([engine() for _ in t])

    k = 30
    selected = np.transpose([[t[k * i], y[k * i]] for i in range(ntrain)])
    trains = selected[0]
    yscale = selected[1, 0]
    ytrain = selected[1] / yscale

    for n in range(2):
        nrow = 0
        ncol = 0

        if n == 1:
            gpr.training_pts.clear()
            gpr._kernel_cache.clear()
            gpr._inv_kernel_cache.clear()
            opt_res = kernel.optimize_hyperparameters(trains, ytrain, True if engine.noise > 0 else False)
            if opt_res is not None:
                gpr.sigma_noise = opt_res[2]

        for i in range(ntrain):
            gpr.add_known(trains[i], ytrain[i])
            stats_mean, stats_sd = gpr.sample_all(t)
            stats_mean = stats_mean.flatten()

            axlst[n][nrow, ncol].plot(t, y, c='red')
            axlst[n][nrow, ncol].plot(t, stats_mean * yscale, c='black', ls='--')
            axlst[n][nrow, ncol].fill_between(t, (stats_mean + 2 * stats_sd) * yscale,
                                              (stats_mean - 2 * stats_sd) * yscale,
                                              fc='silver', ec='grey', ls='--')
            axlst[n][nrow, ncol].scatter([*gpr.training_pts], np.array([*gpr.training_pts.values()]) * yscale,
                                         c='darkred', zorder=100)
            axlst[n][nrow, ncol].set_xlabel('x')
            axlst[n][nrow, ncol].set_ylabel('f(x)')

            nrow += 1
            if nrow > 3:
                nrow = 0
                ncol += 1

    plt.show()


def test_unscaled_exp_kernel():
    """ Tests the exp kernel. """

    import matplotlib.pyplot as plt
    import numpy as np
    from core.expkernel import ExpKernel
    from testsworkdir.toyopti import ToyOptimizer
    from testsworkdir.sekernel import _SEKernel
    from core.gpr import GaussianProcessRegression


    fig0, axlst0 = plt.subplots(4, 2, figsize=(20, 20))
    fig1, axlst1 = plt.subplots(4, 2, figsize=(20, 20))
    axlst = (axlst0, axlst1)

    kernel0 = ExpKernel(alpha=0.1,
                        beta=5.0)
    kernel1 = _SEKernel(50, 1)
    engine = ToyOptimizer(init=1e9,
                          min_=65e7,
                          steps_to_conv=100,
                          noise=0.15,
                          elitism=False,
                          restart_chance=0.25)
    gpr = [GaussianProcessRegression(kernel=kernel,
                                     dims=1,
                                     sigma_noise=0.3,
                                     mean=0,
                                     cache_results=True) for kernel in [kernel0, kernel1]]

    ntrain = 8

    t = np.arange(0, 500)
    y = np.array([engine() for _ in t])

    k = 30
    selected = np.transpose([[t[k * i], y[k * i]] for i in range(ntrain)])
    trains = selected[0]
    ytrain = selected[1]
    yscale = np.std(selected[1])
    ymove  = np.mean(selected[1])

    print(np.mean(ytrain))
    ytrain -= ymove
    print(np.mean(ytrain))

    print(np.var(ytrain))
    ytrain /= yscale
    print(np.var(ytrain))

    # gpr[0].kernel.rho = yscale

    # opt_gam = True
    # opt_res = gpr[0].kernel.optimize_hyperparameters(trains, ytrain, opt_gam)
    # if opt_gam:
    #     gpr[0].sigma_noise = opt_res[2]

    for n in range(2):
        nrow = 0
        ncol = 0

        for i in range(ntrain):
            gpr[n].add_known(trains[i], ytrain[i])
            stats_mean, stats_sd = gpr[n].sample_all(t)
            stats_mean = stats_mean.flatten()

            axlst[n][nrow, ncol].plot(t, y, c='red')
            axlst[n][nrow, ncol].plot(t, stats_mean * yscale + ymove, c='black', ls='--')
            axlst[n][nrow, ncol].fill_between(t, (stats_mean + 2 * stats_sd) * yscale + ymove,
                                              (stats_mean - 2 * stats_sd) * yscale + ymove,
                                              fc='silver', ec='grey', ls='--')
            axlst[n][nrow, ncol].scatter([*gpr[n].training_pts], np.array([*gpr[n].training_pts.values()]) * yscale + ymove,
                                         c='darkred', zorder=100)
            axlst[n][nrow, ncol].set_xlabel('x')
            axlst[n][nrow, ncol].set_ylabel('f(x)')

            nrow += 1
            if nrow > 3:
                nrow = 0
                ncol += 1

    plt.show()


def test_log_exp_kernel():
    """ Tests the exp kernel. """

    import matplotlib.pyplot as plt
    import numpy as np
    from core.expkernel import ExpKernel
    from testsworkdir.toyopti import ToyOptimizer
    from core.gpr import GaussianProcessRegression

    fig, ax = plt.subplots(2, 1, figsize=(20, 20))

    kernel = ExpKernel(alpha=0.5,
                       beta=0.5)
    engine = ToyOptimizer(init=3.2e9,
                          min_=10,
                          steps_to_conv=50,
                          noise=0.001,
                          elitism=False)
    gpr = GaussianProcessRegression(kernel=kernel,
                                    dims=1,
                                    sigma_noise=0.0,
                                    mean=None,
                                    cache_results=True)

    t = np.arange(0, 500)
    y = np.array([engine() for _ in t])

    data = np.transpose([t, y])
    training_data = data[::25]
    t_train, y_train = tuple(np.transpose(training_data))

    y_train = np.log10(y_train)
    y_scale = y_train[0]
    y_train = y_train / y_scale

    opt_res = kernel.optimize_hyperparameters(t_train, y_train, True if engine.noise > 0 else False)
    gpr.sigma_noise = opt_res[2]

    for i in range(len(t_train)):
        gpr.add_known(t_train[i], y_train[i])
    stats_mean, stats_sd = gpr.sample_all(t)
    stats_mean = stats_mean.flatten()

    # LOG SPACE
    ax[0].plot(t, np.log10(y), c='red')
    ax[0].plot(t, stats_mean * y_scale, c='black', ls='--')
    ax[0].fill_between(t, (stats_mean + 2 * stats_sd) * y_scale,
                    (stats_mean - 2 * stats_sd) * y_scale,
                    fc='silver', ec='grey', ls='--')
    ax[0].scatter(t_train, y_train * y_scale,
               c='darkred', zorder=100)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('Log[f(x)]')

    # REAL SPACE
    c = 200
    ax[1].plot(t[c:], y[c:], c='red')
    ax[1].plot(t[c:], np.power(10, stats_mean * y_scale)[c:], c='black', ls='--')
    ax[1].fill_between(t[c:], np.power(10, (stats_mean + 2 * stats_sd) * y_scale)[c:],
                    np.power(10, (stats_mean - 2 * stats_sd) * y_scale)[c:],
                    fc='silver', ec='grey', ls='--')
    ax[1].scatter(t_train[int(c/25):], np.power(10, y_train * y_scale)[int(c/25):],
               c='darkred', zorder=100)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('f(x)')
    m, s = gpr.estimate_mean()
    arr = np.array([m, m+2*s, m-2*s])
    arr = arr * y_scale
    arr = 10 ** arr
    ax[1].set_title(f'mu={arr[0]} + {arr[1]} - {arr[2]}')

    plt.show()


def test_mean_est():

    import matplotlib.pyplot as plt
    import numpy as np
    from core.expkernel import ExpKernel
    from testsworkdir.toyopti import ToyOptimizer
    from core.gpr import GaussianProcessRegression


    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2)
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_uncert = fig.add_subplot(gs[0, 1])
    ax_gp = fig.add_subplot(gs[1, :])

    kernel = ExpKernel(alpha=0.7,
                       beta=5)
    engine = ToyOptimizer(init=None,
                          min_=None,
                          steps_to_conv=None,
                          noise=0.03,
                          elitism=False)
    gpr = GaussianProcessRegression(kernel=kernel,
                                    dims=1,
                                    sigma_noise=0.005,
                                    mean=None,
                                    cache_results=False)

    ntrain = 80

    t = np.arange(0, 500)
    y = np.array([engine() for _ in t])

    k = 2
    selected = np.transpose([[t[k * i], y[k * i]] for i in range(ntrain)])
    trains = selected[0]
    yscale = selected[1, 0]
    ytrain = selected[1] / yscale

    mu = []
    mu_sig = []

    res = kernel.optimize_hyperparameters(trains, ytrain, True if engine.noise > 0 else False)
    if res is not None:
        gpr.sigma_noise = res[2]

    for n in range(ntrain):
        gpr.add_known(trains[n], ytrain[n])
        mean, var = gpr.estimate_mean()
        mu.append(float(mean) * yscale)
        mu_sig.append(float(np.sqrt(var)) * yscale)

    long_t = range(0, 500)
    stats_mean, stats_sd = gpr.sample_all(long_t)
    stats_mean = stats_mean.flatten()

    ax_mean.scatter(range(ntrain), mu)
    ax_mean.set_title("Approximated Mean")
    ax_mean.set_xlabel("Number of Training Points Added to GP")
    ax_mean.set_ylabel("Mean")

    ax_uncert.scatter(range(ntrain), mu_sig)
    ax_uncert.set_title("Uncertainty on Mean ???")
    ax_uncert.set_xlabel("Number of Training Points Added to GP")
    ax_uncert.set_ylabel("\u03C3")

    ax_gp.plot(t, y, c='red')
    ax_gp.plot(long_t, stats_mean * yscale, c='black', ls='--')
    ax_gp.fill_between(long_t, (stats_mean + 2 * stats_sd) * yscale, (stats_mean - 2 * stats_sd) * yscale,
                       fc='silver', ec='grey', ls='--')
    ax_gp.scatter([*gpr.training_pts], np.array([*gpr.training_pts.values()]) * yscale,
                  c='darkred', zorder=100)
    ax_gp.set_title("Gaussian Process")
    ax_gp.set_xlabel("Epoch")
    ax_gp.set_ylabel("Optimizer Solution")
    ax_gp.axhline(y=engine.min_, ls=':', c='black')
    ax_gp.annotate(f"True Value: {engine.min_:.2f}\n"
                   f"Est. Value: {mu[-1]:.2f}\n"
                   f"f(500 000) = {float(gpr.sample_all([500000])[0][0]) * yscale:.2f} "
                   f"\u00B1 {float(np.sqrt(gpr.sample_all([500000])[1][0])) * yscale:.2f}", (max(long_t), engine.min_))

    plt.show()


def regress_cma_traj():
    import matplotlib.pyplot as plt
    import numpy as np
    from core.expkernel import ExpKernel
    from core.gpr import GaussianProcessRegression

    # GRAPH 1
    training_data = []
    with open("cma_fit.dat") as file:
        lines = file.readlines()
        for line in lines:
            if line[0].isdigit():
                val = line.split(' ')[4]
                training_data.append(float(val))
    epochs = range(len(training_data))
    yscale = training_data[0]
    scaled_data = [dat / yscale for dat in training_data]

    kernel = ExpKernel(alpha=0.1,
                       beta=5.0)
    gpr = GaussianProcessRegression(kernel=kernel,
                                    dims=1,
                                    sigma_noise=0.02,
                                    mean=None,
                                    cache_results=True)
    res = kernel.optimize_hyperparameters(epochs, scaled_data)
    if res is not None:
        gpr.sigma_noise = res[2]
    for i in epochs:
        gpr.add_known(i, scaled_data[i])

    stats_mean, stats_sd = gpr.sample_all(epochs)
    stats_mean = stats_mean.flatten()
    mu, sigma = gpr.estimate_mean()

    # GRAPH 2
    training_data2 = []
    with open("history.dat") as file:
        lines = file.readlines()
        for line in lines:
            val = float(line.split(' ')[-2])
            if np.isfinite(val):
                training_data2.append(val)
    epochs2 = range(len(training_data2))
    yscale2 = training_data2[0]
    scaled_data2 = [dat / yscale2 for dat in training_data2]

    kernel2 = ExpKernel(alpha=0.1,
                        beta=5.0)
    gpr2 = GaussianProcessRegression(kernel=kernel2,
                                     dims=1,
                                     sigma_noise=0.01,
                                     mean=None,
                                     cache_results=True)
    res = kernel2.optimize_hyperparameters(epochs2, scaled_data2, x0=(0.3, 0.5, 5.0), bounds=((0.1, 2),
                                                                                              (0.1, 5),
                                                                                              (0.01, 0.2)))
    if res is not None:
        gpr2.sigma_noise = res[2]
    for i in epochs2:
        if not np.isnan(scaled_data2[i]) and not np.isinf(scaled_data2[i]):
            gpr2.add_known(i, scaled_data2[i])

    stats_mean2, stats_sd2 = gpr2.sample_all(np.array(epochs2)+0.5)
    stats_mean2 = stats_mean2.flatten()
    mu2, sigma2 = gpr2.estimate_mean()

    fig, ax = plt.subplots(2, 1, figsize=(20, 20))
    ax[0].scatter(epochs, training_data, c='red', zorder=50)
    ax[0].plot(epochs, stats_mean * yscale, c='black', ls='--')
    ax[0].fill_between(epochs, (stats_mean + 2 * stats_sd) * yscale, (stats_mean - 2 * stats_sd) * yscale,
                       fc='silver', ec='grey', ls='--')
    ax[0].set_title(f"Gaussian Process (\u03BC = {mu * yscale:.2E} \u00B1 {np.sqrt(sigma) * yscale:.2E})")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Optimizer Solution")

    ax[1].plot(epochs2, training_data2, c='red', zorder=50)
    ax[1].plot(epochs2, stats_mean2 * yscale2, c='black', ls='--')
    ax[1].fill_between(epochs2, (stats_mean2 + 2 * stats_sd2) * yscale2, (stats_mean2 - 2 * stats_sd2) * yscale2,
                       fc='silver', ec='grey', ls='--')
    ax[1].set_title(f"Gaussian Process (\u03BC = {mu2 * yscale2:.2E}"
                    f" \u00B1 {np.sqrt(sigma2) * yscale2:.2E})")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Optimizer Solution")

    plt.show()


def regress_real_results():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from core.expkernel import ExpKernel
    from core.gpr import GaussianProcessRegression

    streams = {}
    for file in os.listdir('inputs/ToonOptimizerTests'):
        if file.endswith('.txt'):
            streams[file.replace('.txt', '')] = np.loadtxt(f"inputs/ToonOptimizerTests/{file}")

    for stream in streams:
        kernel = ExpKernel(alpha=0.1,
                           beta=5.0)
        gpr = GaussianProcessRegression(kernel=kernel,
                                        dims=1,
                                        sigma_noise=0.01)

        y = streams[stream]
        yscale = y[0]
        y /= yscale
        t = np.arange(len(streams[stream]))
        step = int(np.max(t) / 50)
        data = np.transpose([t, y])
        t_training, y_training = np.transpose(data[::step])
        for pt in range(len(t_training)):
            gpr.add_known(t_training[pt], y_training[pt])

        # gpr.sigma_noise = kernel.optimize_hyperparameters(t_training, y_training)[2]
        mu, sigma = gpr.estimate_mean()
        stats_mean, stats_sd = gpr.sample_all(t)
        stats_mean = stats_mean.flatten()

        fig, ax = plt.subplots()
        ax.set_title(f"{stream} (\u03BC = {mu * yscale:.2E} \u00B1 {np.sqrt(sigma) * yscale:.2E})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.scatter(t_training, y_training * yscale, c='darkred', zorder=50)
        ax.plot(t, y * yscale, c='red', ls='-')
        ax.plot(t, stats_mean * yscale, c='black', ls='--')
        ax.fill_between(t, (stats_mean + 2 * stats_sd) * yscale, (stats_mean - 2 * stats_sd) * yscale,
                        fc='silver', ec='grey', ls='--')
    plt.show()


def regress_unscaled_real_results():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from core.expkernel import ExpKernel
    from core.gpr import GaussianProcessRegression

    streams = {}
    for file in os.listdir('inputs/ToonOptimizerTests'):
        if file.endswith('.txt'):
            streams[file.replace('.txt', '')] = np.loadtxt(f"inputs/ToonOptimizerTests/{file}")

    for stream in streams:
        kernel = ExpKernel(alpha=0.1,
                           beta=5.0)
        gpr = GaussianProcessRegression(kernel=kernel,
                                        dims=1,
                                        sigma_noise=0.01)

        y = streams[stream]
        t = np.arange(len(streams[stream]))
        step = int(np.max(t) / 50)
        data = np.transpose([t, y])
        t_training, y_training = np.transpose(data[::step])
        for pt in range(len(t_training)):
            gpr.add_known(t_training[pt], y_training[pt])

        # gpr.sigma_noise = kernel.optimize_hyperparameters(t_training, y_training)[2]
        mu, sigma = gpr.estimate_mean()
        stats_mean, stats_sd = gpr.sample_all(t)
        stats_mean = stats_mean.flatten()

        fig, ax = plt.subplots()
        ax.set_title(f"{stream} (\u03BC = {mu:.2E} \u00B1 {np.sqrt(sigma):.2E})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.scatter(t_training, y_training, c='darkred', zorder=50)
        ax.plot(t, y, c='red', ls='-')
        ax.plot(t, stats_mean, c='black', ls='--')
        ax.fill_between(t, (stats_mean + 2 * stats_sd), (stats_mean - 2 * stats_sd),
                        fc='silver', ec='grey', ls='--')
    plt.show()


def regress_log_real_results():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from core.expkernel import ExpKernel
    from core.gpr import GaussianProcessRegression

    streams = {}
    for file in os.listdir('inputs/ToonOptimizerTests'):
        if file.endswith('.txt'):
            streams[file.replace('.txt', '')] = np.loadtxt(f"inputs/ToonOptimizerTests/{file}")

    for stream in streams:
        kernel = ExpKernel(alpha=0.1,
                           beta=5.0)
        gpr = GaussianProcessRegression(kernel=kernel,
                                        dims=1,
                                        sigma_noise=0.01)

        y = streams[stream]
        y = np.log(y)
        yscale = y[0]
        y /= yscale
        t = np.arange(len(streams[stream]))
        step = int(np.max(t) / 50)
        data = np.transpose([t, y])
        t_training, y_training = np.transpose(data[::step])
        for pt in range(len(t_training)):
            gpr.add_known(t_training[pt], y_training[pt])

        gpr.sigma_noise = kernel.optimize_hyperparameters(t_training, y_training)[2]
        mu, sigma = gpr.estimate_mean()
        mu = np.exp(mu * yscale)
        sigma = np.exp(np.sqrt(sigma) * yscale)
        stats_mean, stats_sd = gpr.sample_all(t)
        stats_mean = stats_mean.flatten()

        fig, ax = plt.subplots()
        ax.set_title(f"{stream} (\u03BC = {mu:.2E} \u00B1 {sigma:.2E})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")

        ax.scatter(t_training, np.exp(y_training * yscale), c='darkred', zorder=50)
        ax.plot(t, np.exp(y * yscale), c='red', ls='-')
        ax.plot(t, np.exp(stats_mean * yscale), c='black', ls='--')
        ax.fill_between(t, np.exp((stats_mean + 2 * stats_sd) * yscale), np.exp((stats_mean - 2 * stats_sd) * yscale),
                        fc='silver', ec='grey', ls='--')
    plt.show()


def test_killing_logic():
    import matplotlib.pyplot as plt
    import numpy as np
    import multiprocessing as mp
    from scope.scope import ParallelOptimizerScope
    from time import time
    from core.expkernel import ExpKernel
    from testsworkdir.toyopti import ToyOptimizer
    from core.gpr import GaussianProcessRegression

    def unspool_optimizer(func: ToyOptimizer, runtime, queue_):
        start = time()
        count = 0
        name = func.sec_per_eval
        while time() - start < runtime:
            call = func()
            queue_.put((name, count, call))
            count += 1

    def optimize_hyperparms(title_, kernel, queue_, time_series, loss_series, kwargs: dict):
        res = kernel.optimize_hyperparameters(time_series, loss_series, **kwargs)
        queue_.put((title_, *res))

    def hunt_for_weaklings(gprs_, queue_, scale):
        for main_title in gprs_:
            mu_, sigma_ = gprs_[main_title].sample_all(300)
            sigma_ = sigma_ if not np.isnan(sigma_) else 0
            for comp_title in gprs_:
                if not main_title == comp_title:
                    pt = np.amin([*gprs_[comp_title].training_pts.values()])
                    if pt * scale[comp_title] < (mu_ - 2 * sigma_) * scale[main_title]:
                        queue_.put(main_title)

    manager = mp.Manager()
    queue = manager.Queue()
    opt_queue = manager.Queue()
    hunt_queue = manager.Queue()
    delays = (0.01, 0.02, 0.03)

    plt.ion()
    scope = ParallelOptimizerScope(len(delays), x_range=(0, 300),
                                   record_movie=True, movie_args={'outfile': 'movie.mp4', 'dpi': 200})
    scope.fig.fig_size = (20, 20)
    scope.ax.set_title("Recording of Parallel Optimization Runs")
    scope.ax.set_xlabel("Iteration")
    scope.ax.set_ylabel("Error")

    engines = {t: ToyOptimizer(init=None,
                               min_=None,
                               steps_to_conv=None,
                               noise=0.05,
                               elitism=False,
                               restart_chance=0.25,
                               sec_per_eval=t) for i, t in enumerate(delays)}
    kernels = {t: ExpKernel(0.1, 5.0) for t in delays}
    gprs = {t: GaussianProcessRegression(kernel=kernels[t],
                                         dims=1,
                                         sigma_noise=0.01) for t in delays}
    counter = {t: 0 for t in delays}
    yscale = {t: 0 for t in delays}

    processes = [mp.Process(target=unspool_optimizer, args=(engine, 3 * 60, queue))
                 for engine in engines.values()]
    [p.start() for p in processes]
    while any([p.is_alive() for p in processes]):
        if not queue.empty():
            title, x_pt, y_pt = queue.get_nowait()
            index = delays.index(title)
            if processes[index].is_alive():  # Must do this check in case a killed process still has data in queue
                scope.update_optimizer(index, (x_pt, y_pt))
                if counter[title] % 10 == 0:
                    if counter[title] == 0:
                        yscale[title] = y_pt
                    gprs[title].add_known(x_pt, y_pt / yscale[title])
                    # Optimize after every ten training points
                    if counter[title] % 100 == 0 and counter[title] > 0:
                        scope.update_opt_start(index, (x_pt, y_pt))
                        t_series = np.array([*gprs[title].training_pts])
                        y_series = np.array([*gprs[title].training_pts.values()])
                        opt_proc = mp.Process(target=optimize_hyperparms, args=(title, kernels[title], opt_queue,
                                                                                t_series, y_series,
                                                                                {'bounds': ((0.05, 0.10),
                                                                                            (5, 10),
                                                                                            (0.0001, 0.01))}))
                        opt_proc.start()
                    else:
                        scope.update_scatter(index, (x_pt, y_pt))
                    # Start a hunt after every 50 iterations
                    if counter[title] % 50 == 0 and counter[title] > 0:
                        living = {}
                        for i, p in enumerate(processes):
                            tit = delays[i]
                            if p.is_alive():
                                living[tit] = gprs[tit]
                        hunt = mp.Process(target=hunt_for_weaklings, args=(living, hunt_queue, yscale))
                        hunt.start()
                    mu, sigma = gprs[title].sample_all(300)
                    scope.update_mean(index, mu * yscale[title], sigma * yscale[title])
                    # x = np.arange(700)
                    # mu, sigma = gprs[title].sample_all(x)
                    # mu = mu.flatten()
                    # lower_sig = (mu - 2 * sigma) * yscale[title]
                    # upper_sig = (mu + 2 * sigma) * yscale[title]
                    # mu = mu * yscale[title]
                    # scope.update_gpr(index, x, mu, lower_sig, upper_sig)
                counter[title] += 1

        # Check if any optimisation jobs are done and can be used to update hyperparameters
        if not opt_queue.empty():
            title, alpha, beta, sigma = opt_queue.get_nowait()
            kernels[title].alpha, kernels[title].beta = alpha, beta
            gprs[title].sigma_noise = sigma

            index = delays.index(title)
            x_pt, y_pt = scope.get_farthest_pt(index)

            scope.update_opt_end(index, (x_pt, y_pt))
            mu, sigma = gprs[title].sample_all(300)
            scope.update_mean(index, mu * yscale[title], sigma * yscale[title])
            # x = np.arange(700)
            # mu, sigma = gprs[title].sample_all(x)
            # mu = mu.flatten()
            # lower_sig = (mu - 2 * sigma) * yscale[title]
            # upper_sig = (mu + 2 * sigma) * yscale[title]
            # mu = mu * yscale[title]
            # scope.update_gpr(index, x, mu, lower_sig, upper_sig)

        # Check if any hunting jobs have recommended the termination of an optimizer
        if not hunt_queue.empty():
            title = hunt_queue.get_nowait()
            index = delays.index(title)
            processes[index].terminate()
            processes[index].join()
            scope.update_kill(index)

    [p.join() for p in processes]
    scope.generate_movie()
    print("Done")
    plt.show(block=True)


def test_glompoclass():
    from scm.params.optimizers.cma import CMAOptimizer

    cma = CMAOptimizer()
    print(cma)


def test_toyengines():
    from testsworkdir.toy_engines.rosenbrock import Rosenbrock
    from testsworkdir.toy_engines.ackley import Ackley
    from testsworkdir.toy_engines.expproblem import ExpProblem
    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(1, 30):
        rose = Rosenbrock(i, 0)
        assert rose([1]*i) == 0

    ack = Ackley()
    assert ack(0, 0) == 0

    fig, ax = plt.subplots(1, 3)

    ax[0].set_title('Rosenbrock')
    rose = Rosenbrock(2)
    x = np.linspace(-1.5, 2, 50)
    y = np.linspace(-0.5, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = rose([X, Y])
    ax[0].contourf(X, Y, Z)

    ax[1].set_title('Ackley')
    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    Z = ack(X, Y)
    ax[1].contourf(X, Y, Z)

    ax[2].set_title('ExpProblem')
    exp = ExpProblem(delay=0, npar=2)
    xrange = np.linspace(0.5, 1, 80)
    yrange = np.linspace(0.5, 1, 80)
    X, Y = np.meshgrid(xrange, yrange)
    Z = np.zeros_like(X)
    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            Z[i, j] = exp(np.array([x, y]))
    ax[2].contourf(X, Y, Z)

    plt.show()


def test_datadraw():
    from testsworkdir.toy_engines.data import DataDrawer

    eng = DataDrawer("toy_engines/hist.dat")
    for i in range(20):
        print(eng(3))


def test_signaller():
    import multiprocessing as mp
    from time import time, sleep
    from typing import Type

    class Opt:
        def __init__(self, pipe, queue):
            self.pipe = pipe
            self.queue = queue

        def start(self, x):
            self.pipe.send("Starting")
            for i in range(3):
                print(f"{i} Working")
                self.queue.put(i ** x)
            self.pipe.send("Ended")

    par_pipe, child_pipe = mp.Pipe()
    man = mp.Manager()
    q = man.Queue()
    opt = Opt(child_pipe, q)
    opt.start(2)
    # p = mp.Process(target=opt.start, args=(5,))
    # p.start()
    print(par_pipe.recv())
    print(par_pipe.recv())
    while not q.empty():
        print(q.get_nowait())
    print("Done")
    # p.join()


def test_gpr():
    from core.gpr import GaussianProcessRegression
    from core.expkernel import ExpKernel
    import numpy as np
    import matplotlib.pyplot as plt

    class TestGPR1D:
        gpr = GaussianProcessRegression(kernel=ExpKernel(0.1, 5),
                                        dims=1,
                                        sigma_noise=0,
                                        mean=None,
                                        cache_results=False)
        # yscale = 3.5
        for i in range(10):
            gpr.add_known(i, (0.5 * np.exp(- 0.2 * i) + 3)) #/ yscale)
        y_pts = [(0.5 * np.exp(- 0.2 * i) + 3) for i in range(10)]

        plt.scatter(range(10), [0.5 * np.exp(- 0.2 * i) + 3 for i in range(10)])
        plt.plot(np.linspace(0.5, 20.5, 20), gpr.sample_all(np.linspace(0.5, 20.5, 20))[0])
        plt.plot(np.linspace(0.5, 20.5, 20), (gpr.sample_all(np.linspace(0.5, 20.5, 20))[0] + 2 * gpr.sample_all(np.linspace(0.5, 20.5, 20))[1]))
        plt.plot(np.linspace(0.5, 20.5, 20), (gpr.sample_all(np.linspace(0.5, 20.5, 20))[0] - 2 * gpr.sample_all(np.linspace(0.5, 20.5, 20))[1]))
        print(gpr.sample_all(np.linspace(0.5, 20.5, 20))[1])
        plt.show()

        def test_dict(self):
            for i, x in enumerate(self.gpr.training_pts):
                assert x[0] == i
                assert self.gpr.training_pts[x] * self.yscale == 0.5 * np.exp(- 0.2 * i) + 3

        def test_mean(self):
            before = self.gpr.sample_all(500000)[0] * self.yscale
            self.gpr.mean = 8
            after = self.gpr.sample_all(500000)[0] * self.yscale
            assert not np.isclose(before, after)
            self.gpr.mean = None

        def test_sample_all(self):
            np.random.seed(1)
            x_pts = np.random.rand(25) * 10
            mean, std = self.gpr.sample_all(x_pts)
            assert len(mean) == 25
            assert len(std) == 25
            assert np.ndim(mean) == 1
            assert np.ndim(std) == 1

        def test_sample(self):
            np.random.seed(1)
            x_pts = np.random.rand(25) * 20
            y_pts = self.gpr.sample(x_pts) * self.yscale
            assert len(y_pts) == 25
            assert np.ndim(y_pts) == 1

        def test_estmean(self):
            assert np.isclose(self.gpr.estimate_mean()[0] * self.yscale, 5, atol=1e-2)

        def test_noise(self):
            before = self.gpr.sample_all(np.linspace(0.5, 20.5, 20))[1] * self.yscale
            self.gpr.sigma_noise = 0.01
            after = self.gpr.sample_all(np.linspace(0.5, 20.5, 20))[1] * self.yscale
            assert np.all(before < after)
            self.gpr.sigma_noise = 0

    TestGPR1D().test_sample()


if __name__ == '__main__':
    test_unscaled_exp_kernel()
