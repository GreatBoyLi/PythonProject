_26 = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
secretKey = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def ex_gcd(dividend, divisor):
    x1, x2, x3 = 1, 0, divisor
    y1, y2, y3 = 0, 1, dividend
    while y3 != 0 and y3 != 1:
        q = int(x3 / y3)
        t1, t2, t3 = x1 - q * y1, x2 - q * y2, x3 - q * y3
        x1, x2, x3 = y1, y2, y3
        y1, y2, y3 = t1, t2, t3
    if y3 == 0:
        return -1, x3
    if y3 == 1:
        return y2 % divisor, y3


def encryption(string, num):
    res = ''
    for x in string:
        if 97 <= ord(x) <= 122:
            y = (secretKey.index(x.upper()) * num) % 26
            res += secretKey[y]
        else:
            res += x
    return res


def decryption(string, num):
    res = ''
    for x in string:
        if 65 <= ord(x) <= 90:
            y = (secretKey.index(x) * num) % 26
            res += secretKey[y].lower()
        else:
            res += x
    return res


if __name__ == '__main__':
    over = False
    while not over:
        print('乘法加密')
        print(f'与26互素的数有{_26}')
        key = input('从中选择出乘法加密用到的密钥：')
        a, b = ex_gcd(int(key), 26)
        if b != 1:
            print(f'输入的\"{key}\"没有逆元！')
        message = input('请输入明文：')
        ciphertext = encryption(message, int(key))
        print(f'加密后的密文是：{ciphertext}')
        flag = input('是否需要解密？（Y：解密；N：不解密）')
        if flag == 'Y' or flag == 'y':
            plaintext = decryption(ciphertext, a)
            print(f'解密后的明文是：{plaintext}')
        input_a = input('是否结束？（Y：结束；N；不结束）')
        if input_a == 'Y' or input_a == 'y':
            over = True
        else:
            over = False
