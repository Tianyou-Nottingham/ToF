def longestSubarray( nums: list[int]) -> int:
    maxL = 0
    start = 0
    flag = 0
    for end in range(len(nums)):
        if nums[end] == 0:
            flag += 1
            while flag > 1:
                if nums[start] == 0:
                    flag -= 1
                start += 1
            
            start = end + 1
        maxL = max(maxL, end - start + 1)
    return maxL

if __name__ == "__main__":
    print(longestSubarray([0,1,1,1,0,1,1,0,1])) # 3