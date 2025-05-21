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
        #tb_2 dung de tinh trong ham loss
        tb_2 = tb_1 ** 2
        MSE = 1/2 * TrungBinh(tb_2)
        grad_w0 = TrungBinh(tb_1)
        grad_w1 = TrungBinh(tb_1 * x)

        print(f"Epoch {Epoc}: w0 = {w0:.4f}, w1 = {w1:.4f}, MSE = {MSE:.4f}")

        w0 -= alpha * grad_w0
        w1 -= alpha * grad_w1
        w0_store = np.append(w0_store, w0)
        w1_store = np.append(w1_store, w1)
        loss_store = np.append(loss_store,MSE)
    print(f"ket qua cuoi cung w0 = {w0_store[-2]:.4f} | w1 = {w1_store[-2]:.4f} | MSE = {loss_store[-1]:.4f}")

def compute_mse_single(xi, yi, w0, w1):
    pred = w0 + w1 * xi
    error = pred - yi
    return 0.5 * (error ** 2)
def Stochatic(x, y, epocs, alpha, w0, w1):
    final_losses = []  # Lưu loss cuối cùng của từng điểm

    # Huấn luyện từng điểm một, mỗi điểm lặp 20 epochs
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        print(f"\n--- Huấn luyện cho Data {i + 1}: (x = {xi}, y = {yi}) ---")
        # Epoch 0: trước khi cập nhật
        pred = w0 + w1 * xi
        loss = compute_mse_single(xi, yi, w0, w1)
        print(f"Epoch  0: w0 = {w0:.5f}, w1 = {w1:.5f}, loss = {loss:.5f}")

        for epoch in range(1, epocs + 1):
            # Dự đoán
            pred = w0 + w1 * xi
            error = pred - yi

            # Gradient
            grad_w0 = error
            grad_w1 = error * xi

            # Cập nhật trọng số
            w0 -= alpha * grad_w0
            w1 -= alpha * grad_w1

            # Tính lại loss
            loss = compute_mse_single(xi, yi, w0, w1)
            print(f"Epoch {epoch:2d}: w0 = {w0:.5f}, w1 = {w1:.5f}, loss = {loss:.5f}")

        # Kết quả sau khi train xong điểm hiện tại
        print(f"=> Kết quả sau {epoch} epochs cho Data {i + 1}: w0 = {w0:.5f}, w1 = {w1:.5f}, loss = {loss:.5f}")
        final_losses.append((xi, yi, loss))
    # In kết quả trên toàn bộ dữ liệu sau huấn luyện
    print("\n=== Kết quả cuối cùng trên toàn bộ dữ liệu ===")
    for i, (xi, yi, loss) in enumerate(final_losses):
        print(f"Data {i + 1}: x = {xi}, y = {yi}, loss = {loss:.5f}")


if __name__ == '__main__':
    """
     x = np.array([53,70], dtype='float')
    y = np.array([7,8], dtype='float')
    """
    x = np.array([53, 70, 27, 51, 66, 80, 90, 42], dtype='float')
    y = np.array([7, 8, 3, 6, 8, 10, 12, 5], dtype='float')
    epocs = int(input('nhap so epoc: '))
    alpha = float(input('nhap alpha: '))
    w0_bias = float(input('nhap w0_bias: '))
    w1 = float(input('nhap w1: '))
    #LinearRegression(x,y,epocs,alpha,w0_bias,w1)
    Stochatic(x, y, epocs, alpha, w0_bias, w1)