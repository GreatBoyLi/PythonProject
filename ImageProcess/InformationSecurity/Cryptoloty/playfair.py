crypto_matrix = [[], [], [], [], []]


# 设置密码矩阵
def set_crypto_matrix(cipher_string):
    secretkey = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    secretkey1 = []
    for x in cipher_string:
        if x != ' ' and x.upper() not in secretkey1:
            secretkey1.append(x.upper())
            secretkey.remove(x.upper())
    secretkey1.extend(secretkey)
    secretkey = secretkey1
    print(secretkey)
    secretkey.remove('Z')
    x = 0
    y = 0
    for ch in secretkey:
        if y == 5:
            x = x + 1
            y = 0
        crypto_matrix[x].append(ch)
        y = y + 1
    print(crypto_matrix)


# 判断明文、密文的位置
def get_pos(ch):
    if ch == 'Z':
        return 4, 4
    for x in range(5):
        for y in range(5):
            if ch == crypto_matrix[x][y]:
                return x, y


# 加密
def encryption(string1):
    string_a = list(string1)
    res = ''
    i = 0
    while i < len(string_a):
        if i != len(string_a) - 1:
            if string_a[i] == string_a[i+1]:
                string_a.insert(i+1, 'X')
        elif (i+1) % 2 != 0:
            string_a.append('X')
        x1, y1 = get_pos(string_a[i].upper())
        x2, y2 = get_pos(string_a[i+1].upper())
        i = i + 2
        if x1 == x2:
            y1 = (y1 + 1) % 5
            y2 = (y2 + 1) % 5
            res += crypto_matrix[x1][y1] + crypto_matrix[x2][y2]
        elif y1 == y2:
            x1 = (x1 + 1) % 5
            x2 = (x2 + 1) % 5
            res += crypto_matrix[x1][y1] + crypto_matrix[x2][y2]
        else:
            res += crypto_matrix[x1][y2] + crypto_matrix[x2][y1]
    return res


# 解密
def decryption(string1):
    string_a = list(string1)
    res = ''
    i = 0
    while i < len(string_a):
        if i != len(string_a) - 1:
            if string_a[i] == string_a[i + 1]:
                string_a.insert(i + 1, 'X')
        elif (i + 1) % 2 != 0:
            string_a.append('X')
        x1, y1 = get_pos(string_a[i].upper())
        x2, y2 = get_pos(string_a[i + 1].upper())
        i = i + 2
        if x1 == x2:
            y1 = (y1 - 1) % 5
            y2 = (y2 - 1) % 5
            res += crypto_matrix[x1][y1] + crypto_matrix[x2][y2]
        elif y1 == y2:
            x1 = (x1 - 1) % 5
            x2 = (x2 - 1) % 5
            res += crypto_matrix[x1][y1] + crypto_matrix[x2][y2]
        else:
            res += crypto_matrix[x1][y2] + crypto_matrix[x2][y1]
    return res.lower()


if __name__ == '__main__':
    print('Playfiar密码')
    cipher = input('请输入密钥字母：')
    set_crypto_matrix(cipher)
    message = input('请输入明文：')
    ciphertext = encryption(message.lower())
    print(f'加密后的密文是：{ciphertext}')
    flag = input('是否需要解密？（Y：解密；N：不解密）')
    if flag == 'Y' or flag == 'y':
        plaintext = decryption(ciphertext)
        print(f'解密后的明文是：{plaintext}')
        if 'x' in plaintext:
            print('解密后的明文有x，需要去掉x')
            a = list(plaintext)
            count = a.count('x')
            for i in range(count):
                a.remove('x')
            b = ''
            for ch in a:
                b += ch
            print(f'最终的明文是：{b}')

