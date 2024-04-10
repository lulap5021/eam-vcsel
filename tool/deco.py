from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np




def tmm_smm_waves():
    # n should be a multiple of 3
    n = 600
    x = np.linspace(0, 100, n)

    t = lesin(x) * lexp(x)
    e = lexp(x[0:int(n/3)])
    e3 = np.append(np.append(e, e), e)
    s3 = lesin(x) * e3


    # plot
    fig, ax = plt.subplots()

    ax.plot(x, s3)
    ax.plot(x, 0.5 * np.flip(s3))

    plt.show()



def lesin(x):
    return np.sin(2 * np.pi * 0.12 * x)


def lexp(x):
    return ( np.exp(4e-2 * x) -1 ) / 54


def test_3d():
    palette = ["fec5bb","fcd5ce","fae1dd","f8edeb","e8e8e4","d8e2dc","ece4db","ffe5d9","ffd7ba","fec89a",
               "fbf8cc","fde4cf","ffcfd2","f1c0e8","cfbaf0","a3c4f3","90dbf4","8eecf5","98f5e1","b9fbc0",
               "eddcd2","fff1e6","fde2e4","fad2e1","c5dedd","dbe7e4","f0efeb","d6e2e9","bcd4e6","99c1de"]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    for color in palette:
        r, g, b = string_to_rgb(color)
        rgb_perms = [list(p) for p in permutations([r, g, b])]

        for rgb in rgb_perms:
            r, g, b = rgb
            h, s, l = rgb_to_hsl(r, g, b)
            ax.scatter(h, s, l , c=rgb_to_string(r, g, b))


    plt.show()


def string_to_rgb(color):
    r = int(color[0:2], 16) / 255
    g = int(color[2:4], 16) / 255
    b = int(color[4:6], 16) / 255


    return r, g, b

def rgb_to_string(r, g, b):
    r = int(255 * r)
    g = int(255 * g)
    b = int(255 * b)
    color = '#%02x%02x%02x' % (r, g, b)


    return color

def rgb_to_hsl(r, g, b):
    c = np.max([r, g, b]) - np.min([r, g, b])
    l = (np.max([r, g, b]) + np.min([r, g, b])) / 2
    v = np.max([r, g, b])

    if v == 0:
        s = 0
    else:
        s = c / v

    cmax = np.max([r, g, b])
    if cmax == r:
        h = (g - b) / c % 6
    elif cmax == g:
        h = (b - r) / c + 2
    else:
        h = (r - g) / c + 4

    h = h / 6


    return h, s, l


def hsl_to_rgb(hn, s, l):
    c = (1 - np.abs(2 * l - 1)) * s
    h = 6 * hn
    m = l - c / 2
    x = c *(1 - np.abs(h % 2 -1))

    if 0 <= h < 1:
        r1 = c
        g1 = x
        b1 = 0
    elif 1 <= h < 2:
        r1 = x
        g1 = c
        b1 = 0
    elif 2 <= h < 3:
        r1 = 0
        g1 = c
        b1 = x
    elif 3 <= h < 4:
        r1 = 0
        g1 = x
        b1 = c
    elif 4 <= h < 5:
        r1 = x
        g1 = 0
        b1 = c
    else:
        r1 = c
        g1 = 0
        b1 = x

    r = r1 + m
    g = g1 + m
    b = b1 + m


    return r, g, b

def gen_pastel_hsl(sigma_square=0.2):
    h = np.random.rand()
    s = np.random.rand()
    mu = - 1 * s + 1

    l = -1
    while 0.95 < l or l < 0.75:
        l = 0.25 * np.random.normal(loc=mu, scale=sigma_square) + 0.75


    return h, s, l


def gen_set_pastel_3d(n=500):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(n):
        h, s, l = gen_pastel_hsl()
        r, g, b = hsl_to_rgb(h, s, l)
        ax.scatter(h, s, l, c=rgb_to_string(r, g, b))

    plt.show()