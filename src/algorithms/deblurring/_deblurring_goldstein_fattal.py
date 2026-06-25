import numpy as np
from skimage import io as skio
from skimage.transform import rescale
import matplotlib.pyplot as plt
import tempfile
import scipy
import scipy.io
import scipy.signal
import time
import math
import torch

pi = np.pi
atan2 = math.atan2
sin = np.sin
cos = np.cos
fft = np.fft.fft
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

log = np.log
real = np.real
conj = np.conj
atan2 = math.atan2
tan = np.tan
fftshift = np.fft.fftshift


def RGBtoYCrCb(im, ty="8bit"):
    assert len(im.shape) == 3 and im.shape[2] == 3, "image dans un mauvais format"
    delta = 128
    if ty == "8bit":
        delta = 128  # ce delta est simplement le milieu de la plage des valeurs
        # possibles.
    # out=np.empty(im.shape)
    Y = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    Cr = (im[:, :, 0] - Y) * 0.713 + delta
    Cb = (im[:, :, 2] - Y) * 0.564 + delta
    return (Y, Cr, Cb)


def rgb2gray(u):
    return 0.2989 * u[:, :, 0] + 0.5870 * u[:, :, 1] + 0.1140 * u[:, :, 2]


def viewimage(im, normalize=True, z=1, order=0, titre="", displayfilename=False):
    imin = im.copy().astype(np.float32)
    imin = rescale(imin, z, order=order)
    if normalize:
        imin -= imin.min()
        if imin.max() > 0:
            imin /= imin.max()
    else:
        imin = imin.clip(0, 255) / 255
    imin = (imin * 255).astype(np.uint8)
    filename = tempfile.mktemp(titre + ".png")  # type: ignore
    if displayfilename:
        print(filename)
    plt.imsave(filename, imin, cmap="gray")
    # IPython.display.display(IPython.display.Image(filename))


# alternative viewimage if the other one does not work:
def Viewimage(im, dpi=100, cmap="gray"):
    plt.figure(dpi=dpi)
    if cmap is None:
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.axis("off")
    plt.show()


def YCrCbtoRGB(Y, Cr, Cb, ty="8bit"):
    delta = 128
    out = np.empty([*Y.shape, 3], dtype=np.float32)
    out[:, :, 0] = Y + 1.402 * (Cr - delta)
    out[:, :, 1] = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    out[:, :, 2] = Y + 1.772 * (Cb - 128)
    return out


plot = plt.plot
stem = plt.stem
show = plt.show  # force l'affichage du graphique courant


def norm(X):
    return ((abs(X) ** 2).sum()) ** 0.5


def read_image(fi):
    return np.float32(skio.imread(fi))


# FONCTIONS POUR LA DECONVOLUTION TV


def pad_image(im, pad=10):
    out = np.zeros((im.shape[0] + 2 * pad, im.shape[1] + 2 * pad))
    out[pad:-pad, pad:-pad] = im
    for k in range(pad):
        out[k, pad:-pad] = im[0, :]
        out[-k - 1, pad:-pad] = im[-1, :]
        out[pad:-pad, k] = im[:, 0]
        out[pad:-pad, -k - 1] = im[:, -1]
    out[:pad, :pad] = im[0, 0]
    out[-pad:, :pad] = im[-1, 0]
    out[:pad, -pad:] = im[0, -1]
    out[-pad:, -pad:] = im[-1, -1]
    return out


def unpad_image(im, pad=10):
    return im[pad:-pad, pad:-pad].copy()


def Fourier_kernel(K, s):
    assert K.shape[0] % 2 == 1 and K.shape[1] % 2 == 1, "Taille de noyau impaire"
    Kf = np.zeros(s)
    Ky, Kx = K.shape
    Kx2 = Kx // 2
    Ky2 = Ky // 2
    Kf[: Ky2 + 1, : Kx2 + 1] = K[Ky2:, Kx2:]
    Kf[: Ky2 + 1, -Kx2:] = K[Ky2:, :Kx2]
    Kf[-Ky2:, : Kx2 + 1] = K[:Ky2, Kx2:]
    Kf[-Ky2:, -Kx2:] = K[:Ky2, :Kx2]
    return fft2(Kf)


def taper_image(image, K):
    """Floute une image par le noyau K (circulairement) cela donne une image J
    On mélange l'image avec l'image J de manière à ce que J soit prépondérente aux bords.
    L'image J, lorsqu'on la déconle par le noyau K n'aura pas d'effets de bord."""
    kh, kw = K.shape
    Ih, Iw = image.shape
    wx = np.ones((Ih, Iw), dtype=np.float32)
    wy = np.ones((Ih, Iw), dtype=np.float32)
    X, Y = np.meshgrid(np.arange(0, Iw), np.arange(0, Ih))
    wy[:kh, :] = sin(Y[:kh, :] * pi / (2 * kh - 1)) ** 2
    wy[-kh:, :] = sin((Ih - Y[-kh:, :]) * pi / (2 * kh - 1)) ** 2
    wx[:, :kw] = sin(X[:, :kw] * pi / (2 * kh - 1)) ** 2
    wx[:, -kw:] = sin((Iw - X[:, -kw:]) * pi / (2 * kh - 1)) ** 2
    fK = Fourier_kernel(K, image.shape)
    J = real(ifft2(fft2(image) * fK))
    out = J * (1 - wx * wy) + image * (wx * wy)
    return out


def conv(im, K, Fourierform=False):
    if not Fourierform:  # on recoit les formes spatiales
        Kf = Fourier_kernel(K, im.shape)
        imf = fft2(im)
        return np.real(ifft2(imf * Kf))
    else:  # forme Fourier
        return np.real(ifft2(im * K))


def champ_grad(u):  # gradient circulaire
    return np.stack(
        (
            np.c_[(u[:, 0] - u[:, -1]).reshape(-1, 1), u[:, 1:] - u[:, :-1]],
            np.r_[(u[0, :] - u[-1, :]).reshape(1, -1), u[1:, :] - u[:-1, :]],
        )
    )


def universal_dot(X, Y):
    return (X * Y).sum()


def div_champ(c):
    return (
        np.c_[c[0, :, 1:] - c[0, :, :-1], (c[0, :, 0] - c[0, :, -1]).reshape(-1, 1)]
        + np.r_[c[1, 1:, :] - c[1, :-1, :], (c[1, 0, :] - c[1, -1, :]).reshape(1, -1)]
    )


def d_sub_problem(u, b, gamma=5 / 255):
    gradu = champ_grad(u)
    champ = gradu + b
    s = champ.shape[1:]
    no = (champ**2).sum(axis=0) ** 0.5
    mask = no < (1 / gamma)
    no[mask] = 0.001
    no = no.reshape(1, *s)
    mu = 1 - 1 / (gamma * no)
    champ *= mu
    champ[:, mask] = 0
    # champ[1,mask]=0
    return champ


def u_sub_problem(f, d, b, K, lamb, gamma=5, Fourierform=False, fdenom=None):
    """si Fourierform=True alors f et K sont donnees sous forme Fourier"""
    if not Fourierform:
        ff = fft2(f)
        Kf = Fourier_kernel(K, f.shape)
    else:
        ff = f
        Kf = K
    if fdenom is None:
        Kl = np.zeros(f.shape)
        Kl[0, 0] = 4
        Kl[0, 1] = -1
        Kl[1, 0] = -1
        Kl[-1, 0] = -1
        Kl[0, -1] = -1
        fdenom = real(fft2(Kl))
        fdenom += (lamb / gamma) * (abs(Kf) ** 2)

    numer = conj(Kf) * ff * (lamb / gamma) - fft2(div_champ(d - b))
    return real(ifft2((numer) / fdenom))


def sym_image(x):
    out = np.concatenate((x, np.fliplr(x)), axis=1)
    out = np.concatenate((out, np.flipud(out)), axis=0)  # symetrise l'image
    return out


def TV(im):
    g = champ_grad(im)
    n = ((g**2).sum(axis=0)) ** 0.5
    return n.sum()


def fonctionnelle(f, u, K, d, b, lamb, gamma=5):
    v1 = TV(u) + lamb / 2 * norm(f - conv(u, K)) ** 2
    v2 = v1 + gamma / 2 * norm(d - champ_grad(u) - b) ** 2
    return (v1, v2)


def TVdeconv(im, K, lamb, nbit=140, gamma=5 / 255, edgehandle="taper"):
    """
    Si edgehandle= 'taper' alors on ajoute à l'image une bordure lisse
    Si edgehandle= 'sym' alors on symmetrise l'image
    Si edgehandle= 'nothing' alors on ne fait rien (mauvais)
    """
    if edgehandle == "taper":
        f = taper_image(pad_image(im, K.shape[0]), K)
    elif edgehandle == "sym":
        f = sym_image(im)
    else:
        f = im.copy()
    s = f.shape
    Kf = Fourier_kernel(K, s)
    Kl = np.zeros(f.shape)
    Kl[0, 0] = 4
    Kl[0, 1] = -1
    Kl[1, 0] = -1
    Kl[-1, 0] = -1
    Kl[0, -1] = -1
    fdenom = real(fft2(Kl))
    fdenom += lamb / gamma * (abs(Kf) ** 2)
    u = np.zeros(s)
    unew = np.zeros(s)
    d = np.zeros((2, *s))
    b = np.zeros((2, *s))
    tol = norm(f) / 1000
    counter = 0
    ff = fft2(f)
    # Kfff=conj(Kf)*ff*(lamb/gamma)
    # print("iteration",counter,' Fonctionnelles=',\
    #         fonctionnelle(f,unew,K,d,b,lamb))

    while counter == 0 or (norm(unew - u) > (0 * tol - 1) and counter < nbit):
        counter += 1
        u = unew
        d = d_sub_problem(u, b, gamma=gamma)

        unew = u_sub_problem(
            ff, d, b, Kf, lamb, gamma=gamma, Fourierform=True, fdenom=fdenom
        )
        b += champ_grad(unew) - d
        # print("iteration",counter,)#' Fonctionnelles=',\
        #     fonctionnelle(f,unew,K,d,b,lamb))

    if edgehandle == "taper":
        out = unpad_image(unew, K.shape[0])
    elif edgehandle == "sym":
        out = unew[: im.shape[0], : im.shape[1]]
    else:
        out = unew

    return out


# SinglePhase Retrieval
# Cet algorithme essaye de trouver un noyau seulement en se
# basant sur la densité de puissance de celui-ci
# Partie SINGLEPHASE_RETRIEVAL
def get_phase_alea(Mh, s=None):
    if s is None:
        s = Mh
    im = np.random.randn(Mh, Mh)
    # im[s:,:]=0
    # im[:,s:]=0
    return np.angle(np.fft.fft2(im))


def from_module_phase(module, phase):
    return np.real(np.fft.ifft2(module * np.exp(complex(0, 1) * phase)))


def SinglePhaseRetrieval(
    module, s, Mh=32, alpha=0.95, beta0=0.75, Ninner=300, known=None
):
    fft2 = np.fft.fft2
    phg = get_phase_alea(Mh, s)
    modg = module

    if known is None:
        known = np.ones((Mh, Mh)) > 0
    unknown = (np.vectorize(lambda x: not x))(known)
    modg[unknown] = 0
    g = from_module_phase(modg, phg)

    for m in range(Ninner):
        beta = beta0 + (1 - beta0) * (1 - np.exp(-((m / 7) ** 3)))
        module_for_reconst = modg.copy()
        module_for_reconst[known] = (alpha * module + (1 - alpha) * modg)[known]
        gp = from_module_phase(module_for_reconst, phg)
        mask = 2 * gp < g
        mask[s:, :] = True
        mask[:, s:] = True
        g[mask] = beta * g[mask] + (1 - 2 * beta) * gp[mask]
        invmask = (np.vectorize(lambda x: not x))(mask)
        g[invmask] = gp[invmask]
        fg = fft2(g)
        phg = np.angle(fg)
        modg = abs(fg)
    g[g < 0] = 0
    g[s:, :] = 0
    g[:, s:] = 0
    g = g / g.sum()
    g[g < (1 / 255)] = 0
    g /= g.sum()
    return g[:s, :s]


# FONCTIONS CENTRALES DE L'ESTIMATION DU NOYAU DE FLOU
# Créées suivant <https://www.ipol.im/pub/art/2018/211/>
def Dx_Dy(im):
    """Renvoie deux images de la même taille que im. im est une image en
    niveaux de gris.
    Ce sont les dérivées suivant x et suivant y en utilisant le noyau
    dérivateur spécial"""
    d = np.asarray([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840
    d = d.reshape((1, -1))
    Dx = scipy.signal.convolve2d(im, d, mode="same", boundary="symm")  # type: ignore
    Dy = scipy.signal.convolve2d(im, d.T, mode="same", boundary="symm")  # type: ignore
    return (Dx, Dy)


def entre_Mpi2_pi2(x):
    pi = np.pi
    y = x % pi
    if y >= pi / 2:
        y -= pi
    return y


def liste_thetas_depuis_spectre(N):
    """Renvoies la liste des thetas telles que pour tout point du
    spectre discret de taille NxN corresponde un angle"""
    if N % 2 == 0:
        tmp = np.concatenate((np.arange(0, N // 2 + 1), np.arange(-N // 2 + 1, 0)))
    else:
        tmp = np.concatenate((np.arange(0, (N + 1) // 2), np.arange(-(N - 1) // 2, 0)))
    tmp = tmp.astype(int)  # np.int
    X, Y = np.meshgrid(tmp, tmp)  # carte des fréquences
    pi = np.pi
    Xs = X.reshape(-1)
    Ys = Y.reshape(-1)
    lt = []
    c = 0
    for k in range(len(Xs)):
        if Xs[k] >= 0 and np.gcd(Xs[k], Ys[k]) == 1:
            c += 1
            an = math.atan2(Ys[k], Xs[k])
            lt.append(entre_Mpi2_pi2(an))
    lt = np.asarray(list(set(lt)))  # unique
    lt = np.sort(lt)
    surech = 0 * lt
    for k in range(len(lt)):
        an = lt[k]
        if abs(an) < pi / 4:
            surech[k] = 1 / cos(abs(an))
        else:
            surech[k] = 1 / sin(abs(an))
    return lt, surech


def projections_rapide_shear(tab, thetas, demitaille):
    """projette tab sur toutes les droites d'angles dans thetas.
    La sortie est centree autour de la projection du point central de l'image.
    Nous faisons la projection shear: c'est à dire pas la projection orthogonale
    Cela respecte la méthode originale dans l'article
    Les angles sont censés être entre -pi/2 et pi/2
    Mais leur ordre importe peu.
    """

    L = len(thetas)
    cs = cos(thetas)
    ss = sin(thetas)
    ccs = cs.copy()
    css = ss.copy()
    # modification pour tenir compte du shear
    for m in range(len(thetas)):
        t = thetas[m]
        if abs(t) <= pi / 4:
            ccs[m] = 1
            css[m] = tan(t)
        else:
            css[m] = 1
            ccs[m] = cos(t) / sin(t)

    M, N = tab.shape
    T = M + N + 4
    # le pixel central tombe toujours au même endroit
    poscentre = (T) // 2
    kc = M // 2
    lc = N // 2
    out = np.zeros((L, T))
    outs1 = out.reshape(-1)
    vals = np.zeros(L)
    Ls = (T * np.arange(0, L)).astype(int)  # np.int
    for k in range(M):
        print(k / M * 100, "%", end="\r")
        for z in range(N):
            pos = (
                np.round(ccs * (z - lc) + css * (k - kc)).astype(int) + poscentre
            )  # np.int
            if max(pos) > T:
                print("offenders", k, z)
            vals[:] = tab[k, z]
            outs1[Ls + pos] += vals
    return out[:, poscentre - demitaille : poscentre + demitaille + 1]


def projections_rapide_gradient_shear(DDx, DDy, DDxy, DDyx, thetas, demitaille):
    """utile les intercorralations de Dx,Dy avec eux même pour calculer
    les projections des autocorrelations du noyau
    """

    L = len(thetas)
    cs = cos(thetas)
    ss = sin(thetas)
    cs2 = cs**2
    ss2 = ss**2
    csss = cs * ss
    ccs = cs.copy()
    css = ss.copy()
    # modification pour tenir compte du shear
    for m in range(len(thetas)):
        t = thetas[m]
        if abs(t) <= pi / 4:
            ccs[m] = 1
            css[m] = tan(t)
        else:
            css[m] = 1
            ccs[m] = cos(t) / sin(t)

    M, N = DDx.shape
    T = M + N + 4
    # le pixel central tombe toujours au même endroit
    poscentre = (T) // 2
    kc = M // 2
    lc = N // 2

    out = np.zeros((L, T))
    outs1 = out.reshape(-1)
    vals = np.zeros(L)
    Ls = (T * np.arange(0, L)).astype(int)  # np.int
    for k in range(M):
        print(k / M * 100, "%", end="\r")
        for z in range(N):
            pos = (
                np.round(ccs * (z - lc) + css * (k - kc)).astype(int) + poscentre
            )  # np.int
            if max(pos) > T - 1:
                print("offenders", k, z)
            # print("debug", k,l,DDx.shape)
            vals[:] = (
                cs2 * DDx[k, z] + ss2 * DDy[k, z] + csss * (DDxy[k, z] + DDyx[k, z])
            )
            # if vals[0]!=DDy[k,l]:
            #    print('Houston we have a problem',k,l,vals[0],DDy[k,l],cs2[0],ss2[0],csss[0])
            outs1[Ls + pos] += vals
    return out[:, poscentre - demitaille : poscentre + demitaille + 1]


def next_power_2(T):  # renvoie la puissance de deux immédiatement supérieure
    return int(2 ** (np.floor(np.log(T) / np.log(2)) + 1))


def decoupe(X, A):  # decoupe une partie de tableau et la fftshift
    out1 = np.concatenate((X[: A + 1, -A:], X[: A + 1, : A + 1]), axis=1)
    out2 = np.concatenate((X[-A:, -A:], X[-A:, : A + 1]), axis=1)
    out = np.concatenate((out2, out1), axis=0)
    return out


def calcul_correlations_initiales(img, thetas, p):
    """estime l'autocorrelation du noyau à partir du gradient de l'image
    La deconvolution et le filtrage median sont fait ailleurs"""

    (Dx, Dy) = Dx_Dy(img)
    # calcul des trois correlations
    (a, b) = Dx.shape
    A = next_power_2(2 * a)
    B = next_power_2(2 * b)
    fDx = fft2(Dx, (A, B))
    fDy = fft2(Dy, (A, B))
    DDx = real(ifft2(abs(fDx) ** 2))
    DDy = real(ifft2(abs(fDy) ** 2))
    DDxy = real(ifft2(fDx * conj(fDy)))
    DDyx = real(ifft2(fDy * conj(fDx)))
    DDx = decoupe(DDx, 2 * p)
    DDy = decoupe(DDy, 2 * p)
    DDxy = decoupe(DDxy, 2 * p)
    DDyx = decoupe(DDyx, 2 * p)

    out = projections_rapide_gradient_shear(DDx, DDy, DDxy, DDyx, thetas, 2 * p)
    return out


def deconv_intrinsic_blur(corr, alpha=2.1):
    """Deconvole l'autocorrelation de la projection par un flou
    minimal dû à l'optique."""
    _, qp1 = corr.shape
    M = np.zeros((qp1, qp1))
    for k in range(qp1):
        for z in range(qp1):
            M[k, z] = 1 / ((abs(k - z) + 1) ** alpha)
    M /= M[0, :].sum()
    M = np.linalg.inv(M)
    deconvbrute = (M @ corr.T).T
    print("shape de deconvbrut", deconvbrute.shape)
    poscentre = qp1 // 2
    for k in range(deconvbrute.shape[0]):
        if (deconvbrute[k, poscentre - 2 : poscentre + 2].min()) < 0:
            print(
                "failure of deconv",
                k,
                poscentre,
                deconvbrute[k, poscentre - 2 : poscentre + 2],
            )
            deconvbrute[k] = corr[k]
    return deconvbrute


def initial_support_estimation(tab_corrs, centre, thetas, kappa=30):
    tab_interet = tab_corrs[:, centre:]
    sprime = tab_interet.argmin(axis=1)
    s = (tab_interet.shape[1] - 1) * np.ones(tab_interet.shape[0])
    for k in range(tab_interet.shape[0]):
        if sprime[k] < s[k]:
            s[k] = sprime[k]
            for m in range(tab_interet.shape[0]):
                s[m] = min(s[m], sprime[k] + kappa * abs(thetas[m] - thetas[k]))
    return s


def Estimate_h_correlations(tab_corrs, supports):
    """si le support est connu, on met à zéro tout ce qui dépasse.
    on enleve R[s] à tout le monde
    on normalise à somme 1"""
    centre = tab_corrs.shape[1] // 2
    new_corrs = tab_corrs.copy()
    for k in range(new_corrs.shape[0]):
        sint = int(np.round(supports[k]))
        new_corrs[k, :] -= new_corrs[k, centre + sint]
        new_corrs[k, : centre - sint + 1] = 0
        new_corrs[k, centre + sint :] = 0
        new_corrs[new_corrs < 0] = 0
        new_corrs[k, :] /= (new_corrs[k, :]).sum()
    # filtrage median circulaire
    taille = int(np.ceil(new_corrs.shape[0] ** 0.5))
    out = np.zeros(new_corrs.shape)
    # taille=0 # supprier le filtrage
    for k in range(new_corrs.shape[0]):
        if k - taille < 0:
            tabmed = np.concatenate(
                (new_corrs[: k + taille, :], new_corrs[k - taille :, :]), axis=0
            )
        elif k + taille + 1 > new_corrs.shape[0]:
            tabmed = np.concatenate(
                (
                    new_corrs[k - taille :, :],
                    new_corrs[: ((k + taille + 1) % new_corrs.shape[0]), :],
                ),
                axis=0,
            )
        else:
            tabmed = new_corrs[k - taille : k + taille + 1, :]
        out[k, :] = np.median(tabmed, axis=0)
    return out


def Restimation_supports_noyau(h, p, thetas, ratio=0.05):
    """recalcule les autocorrelations
    du noyau a partir d'une nouvelle estiation et recalcule les supports"""

    (a, b) = h.shape
    A = next_power_2(2 * a + 1)
    B = next_power_2(2 * b + 1)
    fh = fft2(h, (A, B))
    autocor = real(ifft2(abs(fh) ** 2))
    autocor = decoupe(autocor, p)
    # affiche(autocor)

    proj = projections_rapide_shear(autocor, thetas, p)
    # affiche(proj)
    centre = p
    idxs = np.arange(p + 1)
    out = np.zeros(thetas.shape[0])
    for k in range(proj.shape[0]):
        ma = proj[k].max()
        mask = proj[k, centre:] > (ratio * ma)
        out[k] = (idxs * mask).max()
    return out


def calcul_indices_passage_corr_power_spectrum_kernel(N, sc, thetas):
    """Calcule des indices tels que
    f=fft_corr[indices]
    donnera dans f le power spectrum de taille NxN en shape (-1)
    du noyau. fft_corr est la trasnformée de Fourier des correlations
    (en shape(-1)).
    sc est un couple
    thetas sont les angles choisis pour la correlation
    Ce calcul d'indices est fait une fois pour toute pour tous les calculs.
    N : fft2 du noyau doit etre de forme carree NxN
    """

    _, w = sc

    if N % 2 == 0:
        tmp = np.concatenate((np.arange(0, N // 2 + 1), np.arange(-N // 2 + 1, 0)))
    else:
        tmp = np.concatenate((np.arange(0, (N + 1) // 2), np.arange(-(N - 1) // 2, 0)))
    [XX, YY] = np.meshgrid(tmp / N, tmp / N)  # les frequences
    # angle=np.empty(XX.shape,dtype=np.float32)
    numligne = np.empty(XX.shape, dtype=int)  # np.int
    posdansligne = np.empty(XX.shape, dtype=int)  # np.int
    indexs = np.zeros(N * N, dtype=int)  # np.int
    for k in range(XX.shape[0]):
        for z in range(XX.shape[1]):
            angle = entre_Mpi2_pi2(atan2(YY[k, z], XX[k, z]))
            # if angle<-pi/2:
            #    angle+=pi
            # elif angle>pi/2:
            #    angle-=pi
            numligne[k, z] = abs(thetas - angle).argmin()
            # if abs(thetas-angle).min()>0.001:
            #    print ('probleme')
            d = (XX[k, z] ** 2 + YY[k, z] ** 2) ** 0.5
            if abs(thetas[numligne[k, z]]) < pi / 4:
                maxd = 1 / cos(abs(thetas[numligne[k, z]]))
            else:
                maxd = 1 / sin(abs(thetas[numligne[k, z]]))
            posdansligne[k, z] = min(int(np.round(d / maxd * w)), w // 2)

            indexs[k * N + z] = numligne[k, z] * w + posdansligne[k, z]
    return indexs, numligne, posdansligne


def spectre_puissance_depuis_corrs(tab_corrs, N, indexs):
    """transforme des correlations en un spectre de puissance
    utilise des indices precalcules"""
    # zero padding
    h, w = tab_corrs.shape
    center = (w - 1) // 2
    tmp = np.zeros((h, N))
    tmp[:, : center + 1] = tab_corrs[:, center:]
    tmp[:, -center:] = tab_corrs[:, :center]
    out = np.zeros((N, N))
    fcorr = fft(tmp)
    # print(norm(np.imag(fcorr))/norm(fcorr))
    outs1 = out.reshape(-1)
    fcorrs1 = fcorr.reshape(-1)
    outs1[:] = np.real(fcorrs1[indexs])
    outs1[outs1 < 0] = 0  # Les puissances sont négatives
    return out


def convol_carre(im, taille):
    """covole tres rapidement contre un carre"""
    im = im.cumsum(axis=0)
    im = im[taille:, :] - im[:-taille, :]
    im = im.cumsum(axis=1)
    im = im[:, taille:] - im[:, :-taille]
    return im


def calcul_variances_patchs(img, taille):
    """Renvoie une image des variances des patchs de taille=taille X taille
    La sortie de cette fonction permet de trouver dez petites zones dans
    l'image sur lesquels tester la deconvolution"""
    imgmoy = convol_carre(img, taille) / (taille**2)
    imgvar = convol_carre(img**2, taille)
    imgvar = imgvar - (imgmoy**2) * (taille**2)
    return imgvar


def Propose_patch_haute_variance(Varianceimage, img, taille):
    h, w = Varianceimage.shape
    xs = np.random.randint(0, high=w, size=10)
    ys = np.random.randint(0, high=h, size=10)
    pos = Varianceimage.reshape(-1)[xs + ys * w].argmax()
    print(xs[pos], ys[pos])
    return img[ys[pos] : ys[pos] + taille, xs[pos] : xs[pos] + taille].copy()


def score_restau(im):  # utilisée pour comparer des resultats de deconvolution
    # entre eux.
    dx = im[:-1, 1:] - im[:-1, :-1]
    dy = im[1:, :-1] - im[:-1, :-1]
    n = ((dx**2) + (dy**2)) ** 0.5
    return n.sum() / (((n**2).sum()) ** 0.5)


# La fonction principale qui utilise toutes les composantes précédentes
def estime_noyau(
    img, p=25, lamb=1500 / 255, Nouter=3, Ntries=30, Ninner=300, verbose=True
):
    t0 = time.time()
    taille_patch = 150
    imgvars = calcul_variances_patchs(img, taille_patch)
    Nspectrenoyau = 4 * p + 1
    thetas, _ = liste_thetas_depuis_spectre(Nspectrenoyau)
    Nthetas = thetas.shape[0]
    indexs, _, _ = calcul_indices_passage_corr_power_spectrum_kernel(
        Nspectrenoyau, (Nthetas, 4 * p + 1), thetas
    )
    # calcul des autocorrelations de projections du gradient suivant theta
    # sur l'axe theta
    cinit = calcul_correlations_initiales(img, thetas, p)
    # Deconvoluer légèrement les autocorrélations pour suppprimer un
    # "flou intrinsèque
    cdeconv = deconv_intrinsic_blur(cinit)
    # Calcul des supports initiaux
    supports = initial_support_estimation(cdeconv, 2 * p, thetas, kappa=30)
    # En déduire les autocorrélations puis le spectre de puissance de h
    hpower = Estimate_h_correlations(cdeconv, supports)
    H2 = spectre_puissance_depuis_corrs(hpower, Nspectrenoyau, indexs)
    if verbose:
        viewimage(fftshift(H2), titre="densite_spectrale_de_puissance")
        g = SinglePhaseRetrieval(H2, p, Mh=Nspectrenoyau, Ninner=Ninner)
        viewimage(g, titre="premier_noyau_estime")
        print("temps totale de la première phase", time.time() - t0)
    # boucle pour affiner le noyau
    # Seul le support induit par le noyau amélioré est utilisé.
    # On suppose que trois itération de cette boucle suffisent pour atteindre
    # l'optimum de ce que peut faire la méthode
    # on va quand meme stocker tous les trois noyaux dans une liste
    gbests = []
    gbest = np.array([])
    for m in range(Nouter):
        t0 = time.time()
        new_corrs = Estimate_h_correlations(cdeconv, supports)
        H2 = spectre_puissance_depuis_corrs(new_corrs, Nspectrenoyau, indexs)
        cbest = None
        P = Propose_patch_haute_variance(imgvars, img, taille_patch)
        for k in range(Ntries):
            if verbose:
                print("boucle numéro", m, "essai", k, "sur ", Ntries)
            g = SinglePhaseRetrieval(abs(H2) ** 0.5, p, Mh=Nspectrenoyau)
            if (p // 2) * 2 == p:
                gdeconv = np.zeros((p + 1, p + 1))
                gdeconv[:p, :p] = g
            else:
                gdeconv = g

            tmpim = TVdeconv(P, gdeconv, lamb, nbit=40)
            c = score_restau(tmpim)

            if verbose:
                print("score", c)
            if cbest is None or c < cbest:
                gbest = g.copy()
                cbest = c
            g = np.fliplr(np.flipud(g))
            if (p // 2) * 2 == p:
                gdeconv = np.zeros((p + 1, p + 1))
                gdeconv[:p, :p] = g
            else:
                gdeconv = g

            tmpim = TVdeconv(P, gdeconv, lamb, nbit=40)
            c = score_restau(tmpim)
            if verbose:
                print("score flip", c)
            if c < cbest:
                gbest = g.copy()
                cbest = c
        supports = Restimation_supports_noyau(gbest, p, thetas, ratio=0.05)
        if verbose:
            viewimage(gbest, titre="tentative" + str(m))
            print("temps total de la boucle numéro", m, "est ", time.time() - t0)
        gbests.append(gbest)
    return gbest, gbests


def centrer_le_noyau(K):
    # opération optionnelle
    p = K.shape[0]
    assert p % 2 == 1, "Le noyau doit être de taille impaire"
    X, Y = np.meshgrid(np.arange(p), np.arange(p))
    xm = int(np.round((X * K).sum()))
    ym = int(np.round((Y * K).sum()))
    Knew = np.zeros(K.shape)
    dx = min(p - 1 - xm, xm)
    dy = min(p - 1 - ym, ym)
    # print (xm,ym,dx,dy)
    Knew[p // 2 - dy : p // 2 + dy + 1, p // 2 - dx : p // 2 + dx + 1] = K[
        ym - dy : ym + dy + 1, xm - dx : xm + dx + 1
    ]
    print("pourcentage de masse perdue", (1 - Knew.sum() / K.sum()) * 100, "%")
    Knew /= Knew.sum()
    return Knew


def deblurring_goldstein_fattal(RGB_image: torch.Tensor):
    """
    Remove blur from an image using the Goldstein-Fattal deconvolution method.

    The image is decomposed into luminance and chrominance channels, deconvolution is applied to the luminance channel via TV regularization, and the channels are recombined. Input and output images are in the range [0, 1].

    Args:
        RGB_image (torch.Tensor): Input RGB image tensor. Expected range [0, 1], can be on CPU or GPU.

    Returns:
        torch.Tensor: Deblurred RGB image tensor in the range [0, 1], on the same device as the input.
    """
    on_gpu = RGB_image.is_cuda

    RGB_image *= 255
    if on_gpu:
        RGB_image_numpy = RGB_image.cpu().numpy()
    else:
        RGB_image_numpy = RGB_image.numpy()

    image, Cr, Cb = RGBtoYCrCb(RGB_image_numpy)
    kernel, _ = estime_noyau(image)
    kernel_centered = centrer_le_noyau(kernel)
    image_deconv = TVdeconv(image, kernel_centered, 1000 / 255)
    image_deconv.clip(min=0, max=255, out=image_deconv)  # couper les vlaeurs hors 0,255
    RGB_image_deconv = YCrCbtoRGB(image_deconv, Cr, Cb)

    res = torch.from_numpy(RGB_image_deconv).float()
    res /= 255.0
    if on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    res = res.to(device)

    return res
