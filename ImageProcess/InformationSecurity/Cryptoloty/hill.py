import numpy as np

secretkey = np.array([[17, 17, 5], [21, 18, 21], [2, 2, 19]])
secretkey_ni = np.zeros((3, 3))

#
# c = np.array([1, 2, 3])
#
# d = np.matmul(a, c)
# print(d)
#
# f = np.matmul(b, d)
# print(f)


# 加密
def encryption(string):
    i = 0
    res = ''
    index = []
    _flag = False
    while i < len(string):
        count = 0
        j = 0
        right = np.zeros(3)
        while count < 3:
            if (len(string) - i) < 3:
                _flag = True
                break
            if 97 <= ord(string[i+j]) <= 122:
                right[count] = ord(string[i+j]) - 97
                count += 1
            else:
                index.append(i+j)
            j += 1

        if _flag:
            break

        i += j
        result = np.matmul(secretkey, right) % 26
        result += 65
        for x in result:
            res += chr(int(x))

    if _flag:
        num = (len(string) - i)
        right = np.zeros(3)
        if num == 2:
            if 97 <= ord(string[-2]) <= 122:
                right[0] = ord(string[-2]) - 97
            else:
                right[0] = ord('z') - 97
                index.append(len(string)-2)

            if 97 <= ord(string[-1]) <= 122:
                right[1] = ord(string[-1]) - 97
            else:
                right[1] = ord('z') - 97
                index.append(len(string) - 1)

            right[2] = ord('z') - 97
        else:
            if 97 <= ord(string[-1]) <= 122:
                right[0] = ord(string[-1]) - 97
            else:
                right[0] = ord('z') - 97
                index.append(len(string) - 1)

            right[1] = ord('z') - 97
            right[2] = ord('z') - 97

        result = np.matmul(secretkey, right) % 26
        result += 65
        for x in result:
            res += chr(int(x))

    res1 = list(res)
    for y in index:
        res1.insert(y, ' ')
    res = ''.join(res1)
    return res


# 解密
def decryption(string):
    i = 0
    res = ''
    index = []
    _flag = False
    while i < len(string):
        count = 0
        j = 0
        right = np.zeros(3)
        while count < 3:
            if (len(string) - i) < 3:
                _flag = True
                break
            if 65 <= ord(string[i + j]) <= 90:
                right[count] = ord(string[i + j]) - 65
                count += 1
            else:
                index.append(i+j)
            j += 1

        if _flag:
            break

        i += j
        result = np.matmul(secretkey_ni, right) % 26
        result += 97
        for x in result:
            res += chr(int(x))

    if _flag:
        num = (len(string) - i)
        right = np.zeros(3)
        if num == 2:
            if 65 <= ord(string[-2]) <= 90:
                right[0] = ord(string[-2]) - 65
            else:
                right[0] = ord('Z') - 65
                index.append(len(string)-2)

            if 65 <= ord(string[-1]) <= 90:
                right[1] = ord(string[-1]) - 65
            else:
                right[1] = ord('Z') - 65
                index.append(len(string) - 1)

            right[2] = ord('Z') - 65
        else:
            if 65 <= ord(string[-1]) <= 90:
                right[0] = ord(string[-1]) - 65
            else:
                right[0] = ord('Z') - 65
                index.append(len(string) - 1)

            right[1] = ord('Z') - 65
            right[1] = ord('Z') - 65

        result = np.matmul(secretkey_ni, right) % 26
        result += 97
        for x in result:
            res += chr(int(x))

    res1 = list(res)
    for y in index:
        res1.insert(y, ' ')
    res = ''.join(res1)
    return res


# 求解密密钥
def getkey():
    global secretkey_ni
    y = 17
    k1 = np.linalg.inv(secretkey)
    k_abs = np.linalg.det(secretkey)
    # 伴随矩阵
    k2 = k1 * k_abs % 26
    k2 = np.around(k2)
    k2 = k2.astype(int)
    secretkey_ni = y * k2 % 26


if __name__ == '__main__':
    over = False
    while not over:
        print('Hill密码')
        message = input('请输入明文：')
        ciphertext = encryption(message)
        print(f'加密后的密文是：{ciphertext}')
        flag = input('是否需要解密？（Y：解密；N：不解密）')
        if flag == 'Y' or flag == 'y':
            getkey()
            plaintext = decryption(ciphertext)
            print(f'解密后的明文是：{plaintext}')
        input_a = input('是否结束？（Y：结束；N；不结束）')
        if input_a == 'Y' or input_a == 'y':
            over = True
        else:
            over = False
