import numpy as np


def TrungBinh(input):
    kq = 0
    for number in range(len(input)):
        kq += input[number]
    return kq / len(input)


def LinearRegression(x, y, epoc, alpha, w0, w1):
    w0_store = np.array([w0], dtype='float')
    w1_store = np.array([w1], dtype='float')
    loss_store = np.array([w1], dtype='float')
    print(f"cac gia tri x: {x}")
    print(f"cac gia tri y: {y}")
    for Epoc in range(epoc+1):
        y_mu = w1 * x + w0
        tb_1 = y_mu - y
        tb_2 = tb_1 ** 2
        MSE = 1/2 * TrungBinh(tb_2)
        grad_w0 = TrungBinh(tb_1)
        grad_w1 = TrungBinh(tb_1 * x)

        print(f"Epoch {Epoc}: w0 = {w0:.5f}, w1 = {w1:.5f}, MSE = {MSE:.6f}")

        w0 -= alpha * grad_w0
        w1 -= alpha * grad_w1
        w0_store = np.append(w0_store, w0)
        w1_store = np.append(w1_store, w1)
        loss_store = np.append(loss_store,MSE)
    print(f"ket qua cuoi cung w0 = {w0_store[-1]:.5f} | w1 = {w1_store[-1]:.5f} | MSE = {loss_store[-1]:.5f}")
    return w0, w1, y_mu


if __name__ == '__main__':
    x = np.array([53, 70, 27, 51, 66, 80, 90, 42], dtype='float')
    y = np.array([7, 8, 3, 6, 8, 10, 12, 5], dtype='float')
    epocs = int(input('nhap so epoc: '))
    alpha = float(input('nhap alpha: '))
    w0_bias = float(input('nhap w0_bias: '))
    w1 = float(input('nhap w1: '))
    LinearRegression(x, y, epocs, alpha, w0_bias, w1)