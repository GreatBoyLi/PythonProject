import rsa
from binascii import b2a_hex, a2b_hex


class RsaCrypt:
    def __init__(self, publickey=None, privateKey=None):
        self.ciphertext = None
        self.publicKey = publickey
        self.privateKey = privateKey

    def encrypt(self, text):
        self.ciphertext = rsa.encrypt(text.encode(), self.publicKey)
        # 因为rsa加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串
        return b2a_hex(self.ciphertext)

    def decrypt(self, text):
        decrypt_text = rsa.decrypt(a2b_hex(text), self.privateKey)
        return decrypt_text.decode()


if __name__ == '__main__':
    pubkey, prikey = rsa.newkeys(3250)
    rs_obj = RsaCrypt(pubkey, prikey)
    array = [i for i in range(100)]
    text1 = str(array)
    ency_text = rs_obj.encrypt(text1)
    print(ency_text)
    print(rs_obj.decrypt(ency_text))
    print(pubkey)
    print(prikey)
