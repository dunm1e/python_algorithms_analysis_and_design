class KMP:
    def next(self, pattern):
        ret = [0]

        for i in range(1, len(pattern)):
            j = ret[i - 1]
            while j > 0 and pattern[j] != pattern[i]:
                j = ret[j - 1]
            ret.append(j + 1 if pattern[j] == pattern[i] else j)
        return ret

    def search(self, main_str, model_str):

        partial, ret, j = self.next(model_str), [], 0
        m_len, p_len = len(main_str), len(model_str)

        i = 0
        while i <= m_len - p_len:
            while j < p_len and main_str[i + j] == model_str[j]:
                j += 1
            if j == p_len:
                ret.append(i)
                i += j # 跳过当前字符串
                j = 0
            elif j == 0:
                i += 1
            else:
                j = partial[j - 1]
                i += 1
        if not ret:
            print(-1)
        return ret

def min(ret):
    if ret == []:
        return
    min  = ret[1]-ret[0]
    for i in range(1,len(ret)-1):
        if ret[i+1]-ret[i]<min:
            min = ret[i+1]-ret[i]
    return min
T = input("主串：")
P = input("模式串：")

ret = KMP().search(T, P)

if ret != []:
    min_len = min(ret)
    print(ret)
    print(min_len)
