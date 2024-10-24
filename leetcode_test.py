def compress(chars: list[str]) -> int:
    L = len(chars)
    i = 0
    start = 0
    while i < L-1:
        nums = 1
        flag = 0
        for j in range(i+1, L):
            if chars[j] == chars[i]:
                flag = 1
                nums += 1
                i = j
                if i == L-1:
                    chars[start+1: start+len(str(nums))+1] = str(nums)
                continue
            else:
                if flag == 0:
                    i = j
                    start = j
                    break
                else:
                    chars[start+1: start+len(str(nums))+1] = str(nums)
                    i = j
                    start = j
                    break
                
if __name__ == "__main__":
    s = ["a", "a", "b", "b", "c", "c", "c"]
    print(compress(s))
