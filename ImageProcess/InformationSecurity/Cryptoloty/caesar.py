# 加密
def encryption(string):
    res = ''
    for x in string:
        if 97 <= ord(x) <= 122:
            y = (ord(x) - 97 + 3) % 26 + 65
            res += chr(y)
        else:
            res += x
    return res


# 解密
def decryption(string):
    res = ''
    for x in string:
        if 65 <= ord(x) <= 90:
            y = (ord(x) - 65 - 3) % 26 + 97
            res += chr(y)
        else:
            res += x
    return res


if __name__ == '__main__':
    print('Casesar密码')
    message = input('请输入明文：')
    ciphertext = encryption(message)
    print(f'加密后的密文是：{ciphertext}')
    flag = input('是否需要解密？（Y：解密；N：不解密）')
    if flag == 'Y' or flag == 'y':
        plaintext = decryption(ciphertext)
        print(f'解密后的明文是：{plaintext}')

