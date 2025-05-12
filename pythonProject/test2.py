import numpy as np


def TrungBinh(input):
    kq = 0
    for number in range(len(input)):
        kq += input[number]
    return kq / len(input)


def LinearRegression(x, y, epoc, alpha, w0, w1):
    w0_store = np.array([w0], dtype='float')
    w1_store = np.array([w1], dtype='float')
    y_mu_store = np.array([], dtype='float')  # Khởi tạo y_mu_store

    print(f"cac gia tri x: {x}")
    print(f"cac gia tri y: {y}")

    for Epoc in range(epoc + 1):
        y_mu = w1 * x + w0
        y_mu_store = y_mu  # Gán y_mu cho y_mu_store

        tb_1 = y_mu_store - y
        tb_2 = tb_1 ** 2
        MSE = 0.5 * TrungBinh(tb_2)
        grad_w0 = TrungBinh(tb_1)
        grad_w1 = TrungBinh(tb_1 * x)

        print(f"Epoch {Epoc}: w0 = {w0:.5f}, w1 = {w1:.5f}, MSE = {MSE:.6f}")

        w0 -= alpha * grad_w0
        w1 -= alpha * grad_w1
        w0_store = np.append(w0_store, w0)
        w1_store = np.append(w1_store, w1)

    print(f"Final y_mu (y_pred): {y_mu_store}")
    return w0, w1, y_mu_store


if __name__ == '__main__':
    x = np.array([53, 70, 27, 51, 66, 80, 90, 42], dtype='float')
    y = np.array([7, 8, 3, 6, 8, 10, 12, 5], dtype='float')
    epocs = int(input('nhap so epoc: '))
    alpha = float(input('nhap alpha: '))
    w0_bias = float(input('nhap w0_bias: '))
    w1 = float(input('nhap w1: '))
    w0, w1, y_pred = LinearRegression(x, y, epocs, alpha, w0_bias, w1)