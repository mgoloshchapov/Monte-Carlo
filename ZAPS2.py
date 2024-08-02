import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


@njit
def ww(z):  # beam radius from z
    return w0 * np.sqrt(1 + ((lamb * z)/(np.pi * w0**2))**2 * 1e-6)  # mkm

@njit
def kinetic(Vx, Vy, Vz):
    return 1.67 * m*(Vx**2 + Vy**2 + Vz**2)/(2 * 1.38) * 1e2

@njit
def pot_energy(x, y, z):  # Potential energy of atom
    return -U0 * w0**2/(ww(z) ** 2) * np.exp(-2 * (x ** 2 + y ** 2) / (ww(z) ** 2))

@njit
def energy(x, y, z, Vx, Vy, Vz):
    return kinetic(Vx, Vy, Vz) + pot_energy(x, y, z)


@njit
def rnd(x, Delta, del_a, df):  # Trial variances of distributions
    # del_a: Variance of the trial distribution (trial_sigma)
    x0 = x[0] + del_a * Delta[0] * np.random.standard_t(df, size=1)[0]
    x1 = x[1] + del_a * Delta[1] * np.random.standard_t(df, size=1)[0]
    x2 = x[2] + del_a * Delta[2] * np.random.standard_t(df, size=1)[0]
    x3 = x[3] + del_a * Delta[3] * np.random.standard_t(df, size=1)[0]
    x4 = x[4] + del_a * Delta[4] * np.random.standard_t(df, size=1)[0]
    x5 = x[5] + del_a * Delta[5] * np.random.standard_t(df, size=1)[0]
    return x0, x1, x2, x3, x4, x5


@njit
def density(x, y, z, Vx, Vy, Vz, T):  # Probability density function (pdf)
    # return 0.0001+np.exp(-1.67/(2 * 1.38 * 1e-2) * m/T * (Vx**2 + Vy**2 + Vz**2)) \
    #     * np.exp(-pot_energy(x, y, z)/T)
     return (np.exp(-(energy(x, y, z, Vx, Vy, Vz))/T))
@njit
def metropolis_hastings(target_density, x0, T, del_a, df, N_m_h):  # generate samples that follow pdf
    # del_a: Trial standard deviation
    burnin_size = 10000
    N_m_h += burnin_size
    xt = x0  # The beginning of the Markov chain
    samples = []
    k = 0
    acc = []
    for i in prange(N_m_h):
        xt_candidate = rnd(xt, Delta(T), del_a, df)
        accept_prob = (target_density(*xt_candidate, T))/(target_density(*xt, T))
        # accept_prob = np.exp(-(energy(*xt_candidate)-energy(*xt))/T)
        if np.random.uniform(0, 1) < accept_prob:
            k += 1
            xt = xt_candidate
        # else:
        #     while np.random.uniform(0, 1) >= accept_prob:
        #         xt_candidate = rnd(xt, Delta(T), del_a, df)
        #         accept_prob = np.exp(-(energy(*xt_candidate) - energy(*xt)) / T)
        #         # accept_prob = (target_density(*xt_candidate, T)) / (target_density(*xt, T))
        #     xt = xt_candidate
        samples.append(xt)
        acc.append(accept_prob)
    print('Accept_prob =', k/N_m_h)  # Probability of acceptance
    samples = np.array(samples[burnin_size::2])  # Prorezhivanie vyborki
    return np.transpose(samples)


@njit
def probs(x, y, z, Vx, Vy, Vz, tau):  # generate arrays of probabilities
    """
    param x, y, z: An array of x, y, z-coordinates
    param Vx, Vy, Vz: An array of Vx, Vy, Vz-velocities
    param tau: An array of times
    """
    p_list = []
    sw = 0
    for t in tau:
        n = 0
        k = 0
        for i in prange(len(x)):
            if (1.67 * m * (Vx[i] ** 2 + Vy[i] ** 2 + Vz[i] ** 2) / 2 <
                        1.38 * 1e-2 * (-1) * pot_energy(x[i], y[i], z[i])):
                k += 1
                if (1.67 * m * (Vx[i] ** 2 + Vy[i] ** 2 + Vz[i] ** 2) / 2 <
                        1.38 * 1e-2 * (-1) * pot_energy(x[i] + Vx[i] * t, y[i] + Vy[i] * t, z[i] + t * Vz[i])):
                    n += 1
        l = len(x)
        if t == 0:
            print(k, l)
        if t == 0 and k < 0.1 * l:
            print('Initially, there are not enough atoms in the trap')
            sw = 1
            break

        p_list.append(n/k)
    return p_list, sw


@njit
def MaxBol(Delta):  # generate arrays with normal distribution
    x = np.random.normal(0, Delta[0], N)
    y = np.random.normal(0, Delta[1], N)
    z = np.random.normal(0, Delta[2], N)
    Vx = np.random.normal(0, Delta[3], N)
    Vy = np.random.normal(0, Delta[4], N)
    Vz = np.random.normal(0, Delta[5], N)
    return x, y, z, Vx, Vy, Vz


@njit
def Delta(T):  # generate variations of arrays with normal distribution
    # oscillation frequencies
    w_z = np.sqrt((2 * 1.38 * U0 * lamb ** 2) / (1.67 * np.pi ** 2 * 87 * w0 ** 4)) * np.sqrt(1e-8)  # MHz
    w_r = np.sqrt((4 * U0 * 1.38) / (1.67 * m * w0 ** 2)) * np.sqrt(1e-2)  # MHz

    # standart devitations
    delta_x = 0.1 * np.sqrt((1.38 * T) / (m * 1.67 * w_r ** 2)) * np.sqrt(1e-2)
    delta_y = delta_x
    delta_z = 0.1 * np.sqrt((1.38 * T) / (m * 1.67 * w_z ** 2)) * np.sqrt(1e-2)


    delta_Vx = np.sqrt((1.38 * T) / (m * 1.67)) * np.sqrt(1e-2)
    delta_Vy = delta_Vx
    delta_Vz = delta_Vx

    Delt = (delta_x, delta_y, delta_z, delta_Vx, delta_Vy, delta_Vz)
    return Delt



def avg_chains(T, del_a,  df, num, norm):

    """
        param T: Temperature
        param del_a: Variance of the trial distribution (trial_sigma)
        param num: Number of averages
        param norm: 0 or 1 - use Metropolis_Hastings or Normal Distribution
    """

    a = []
    for i in prange(num):
        d = Delta(T)
        x0 = tuple(np.transpose(np.array((MaxBol(d))))[0])
        #x0 = 0,0,0,0,0,0
        if norm == 0:
            ar, swit = probs(*metropolis_hastings(density, x0, T, del_a, df, N_m_h), tau)
        else:
            ar, swit = probs(*MaxBol(d), tau)
        if swit != 1:
            a.append(ar)
    a = np.transpose(np.asarray(a))
    b = tuple([np.average(a[i]) for i in range(len(a))])
    return b


def plot_beauty():
    plt.xlim(0, tau[-1])
    plt.ylim(0, 1.1)
    plt.grid(which='major', lw=0.3)
    plt.tick_params(which='major', direction="in")
    plt.xlabel(r'Время, мкс', family='Arial', size=14)
    plt.ylabel(r'Вероятность', family='Arial', size=14)
    plt.suptitle('График зависимости вероятоности атома остаться\n'
                 'в потенциальной яме от времени выключения поля', family='Arial', size=14)


def read_file(name):
    aa = np.loadtxt(name)
    bb = np.transpose(aa)
    #plt.errorbar(bb[0], bb[1] / max(bb[1]), yerr=bb[2] / max(bb[1]), zorder=-2, linestyle="None")
    return np.array(bb[1] / max(bb[1])), bb[0]


def plot_arr(tau, a):  # create graphs
    if len(a) != 0:
        plot_beauty()
        plt.plot(tau, a)
    else:
        print("\nМало атомов изначально, меняйте дисперсию пробного распределения")


def imedia(a, ara, exp, df):  # intermediate function
    avg0 = avg_chains(ara[0], 1, df, 1, 1)
    s0 = sum(((np.array(avg0)-exp) ** 2)[2:])
    if ara[0]-ara[1] == 5:
        a = a - 5
    for i in range(len(ara)-1):
        avg = avg_chains(ara[i+1], 1, df, 1, 1)
        s = sum(((np.array(avg) - exp)**2)[2:])
        if s < s0:
            s0 = s
            a = ara[i+1]
            avg0 = avg
    return avg0, a


def fit(exp, Tmax, df):
    """
        param exp: experimental array
        param Tmax: Maximal temperature
    """
    a = 10
    ara = np.arange(a, Tmax, 10)
    b = imedia(a, ara, exp, df)[1]
    ara = np.arange(b-5, b+5, 5)
    avg0, a = imedia(b, ara, exp, df)
    plot_arr(tau, avg0)
    plt.fill_between(tau, avg_chains(a-15, 1, df, 1, 1), avg_chains(a+15, 1, df, 1, 1), color="red", alpha=0.2)
    return avg0, a


def sravni(T, del_a, df, num, tau):  # compare MH-algorithm and multivariate normal distribution
    """
            param T: Current temperature
            param del_a: Trial standard deviation
            param num: number of averages
            param tau: array of times
        """
    a = avg_chains(T, del_a, df, 3, 1)
    b = avg_chains(T-1, del_a, df, num, 0)
    c = avg_chains(T+1, del_a, df, num, 0)
    plt.figure()
    plot_beauty()
    plot_arr(tau, a)
    plt.fill_between(tau, b, c, color="red", alpha=0.2)


def Gelman_Rubin(T, del_a, df):
    a = []
    av = [[], [], [], [], [], []]
    J = 10
    for i in range(J):
        s = metropolis_hastings(density, tuple(np.transpose(np.array((MaxBol(Delta(T)))))[0]), T, del_a, df, N_m_h)
        #s = MaxBol(Delta(T))
        a.append(s)
        for j in range(6):
            av[j].append(np.average(s[j]))
    a = np.asarray(a)
    av = np.asarray(av)
    L = len(a[0][0])
    star = []
    for i in range(6):
        star.append(np.average(av[i]))
    B = []
    for i in range(6):
        B.append(L/(J-1) * sum((np.asarray(av[i])-star[i])**2))
    W = np.zeros(6)
    for i in range(6):
        for j in range(J):
            W[i] += 1/(L-1) * sum(np.asarray(a[j][i] - av[i][j])**2)
        W[i] *= 1/J
    B = np.asarray(B)
    W = np.asarray(W)
    R = ((L-1)/L * W + 1/L * B)/W
    return R




'''
Ниже мы строим два графика с одинаковыми параметрами. В идеале они должны совадать. Если не совпадают чуть-чуть простительно.
Если не совпадают сильно, меняем del_a (дисперисию пробного распределния, именно эта штука влияет на сходимось цепочки Маркова).
Начинать стоит с del_a = 1. При малых температурах del_a должно быть побольше, при больщих температурах поменьше. 
Может оказаться так, что из-за большой температуры изначально ни один атом не окажется в ловушке, массивы будут пустыми, поэтому
чтобы не было ошибок, стоит блок if - else. 
В консоли выводятся вероятность принятия следующего элемента в цепочке Маркова (Accept_prob) и кол-во атомов, которые изначально
попали в ловушку при данной температуре (При генерации сэмплов с заданным pdf не все атомы удовлетворяют условию нахождения внутри
ямы). Это число должно быть как можно ближе к числу, которое стоит рядом (количестов сгенрированных сэмплов с учетом прореживания)
Accpet_prob должен быть < 0.95. В идеале 0.25 - 0.50. В идеале, графики должны совпадать и accept_prob должен быть в этом 
интервале. Если одно из двух не работает лучше, чтобы хотя бы графики совпадали. При больших температурах, очень часто, 
одновременное выполнение этих требований невозможно. Если появляется надпись: "Initially, there are not enough atoms in the trap",
то это значит, что в ловушку изначально попало очень мало атомов(10 % от максимально возможного числа). В этом случае эти 
значения при усреднении не используются. 
Также, в программе есть возможность при генерации сэмлов вместо алгоритма Метрополиса_Гатсингса использовать нормальное
распределение. До 60 мкК эти два подхода дают одинаковые результаты. При увеличении глубины ловушки максимальная температура, 
при которой данные подходы дают одинаковые результаты, увеличивается, что дает возможность использовать более бытструю генерацию 
из нормального распределения.
Ниже представлена таблица при глубине ловушки U0 = 700 mkK, радиусе перетяжки w0 = 1 mkm, длине волны lamb = 813 nm. 

Temperature          Trial sigma           Accept_prob
10                      1                       0.27    
20                      1                       0.27
30                      1                       0.28
40                      1                       0.29
50                      1                       0.40
60                      0.9                     0.50
70                      0.3                     0.80
80                      0.1                     0.93

'''
# Parameters
w0 = 1.1  # mkm
lamb = 813  # nm
m = 87  # atomic number
N = 200000  # number of samples in normal distribution
N_m_h = 500000  # number of samples in Metropolis-Hastings algorithm
U0 = 700  # mkK

tau = np.linspace(0, 50, 100)
print('wwww')
exp, tau = read_file('Files/2023_09_10_release_recapture.dat')
plot_beauty()  # axis captions
plt.plot(tau, exp, 'blue')
avg, T = fit(exp, 150, 100)
print('Temperature =', T)
plt.plot(tau, avg)
# sravni(150, 1, 4, 10, tau)
# del_a = 1
# df = 4
# T = 100
# s = metropolis_hastings(density, tuple(np.transpose(np.array((MaxBol(Delta(T)))))[0]), T, del_a, df, N_m_h)
# a = MaxBol(Delta(T))
# x, y, z, Vx, Vy, Vz = np.transpose(a)[0]
# print(x, y, z, Vx, Vy, Vz)
# print(pot_energy(x, y, z)/T)
# print(kinetic(Vx, Vy, Vz)/T)
# print(density(x, y, z, Vx, Vy, Vz, T))

plt.show()


