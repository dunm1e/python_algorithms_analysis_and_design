from typing import List

#给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
#示例 1：
#输入：nums = [2,3,1,1,4]
#输出：true
#解释：我们可以先跳 1 步，从下标 0 到达下标 1，然后再从下标 1 跳 3 步到达最后一个下标。
#示例 2：
#输入：nums = [3,2,1,0,4]
#输出：false
#解释：无论怎样，你总会到达索引 3 。
#贪心算法
# def jumpnum(nums:List[int])->bool:
#     n = len(nums)
#     if n == 1:
#         return True
#     count = 0
#     while count<n:
#         if nums[count] == 0:
#             return False
#         count += nums[count]
#     return True


# if __name__ == '__main__':
#     nums = list(map(int, input("请输入一个整数数组：").split(',')))
#     print(jumpnum(nums))


# 摆动排序：给你一个的整数数组 nums, 将该数组重新排序后使 nums[0] <= nums[1] >= nums[2] <= nums[3]... 输入数组总是有一个有效的答案。

#贪心
# def wiggleSort(nums:List[int])->List[int]:
#     nums.sort()
#     nums[::2], nums[1::2] = nums[:(len(nums) + 1) // 2][::-1], nums[(len(nums) + 1)//2:][::-1]
#      # 在偶数下标和奇数下标和数组前半部分和后半部分逆序后插入
#     return nums
   
        
# # 人才思路：先排序，然后交换相邻的两个元素 若改成严格不等则不适用 如 4，5，5，6
# # def wiggleSort(nums:List[int])->List[int]:
# #     nums.sort()
# #     for i in range(1,len(nums)-1,2):
# #         nums[i],nums[i+1] = nums[i+1],nums[i]
# #     return nums
    
# if __name__ == '__main__':
#     nums = list(map(int, input("请输入一个整数数组：").split(',')))
#     print(wiggleSort(nums))

# 加油站：在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
# 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
# 给定两个整数数组 gas 和 cost ，如果你可以按顺序绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。
# def canCompleteCircuit(gas:List[int],cost:List[int])->int:
#     n = len(gas)
#     for i in range(n):
#         if(cost[i]>gas[i]):
#             continue
#         else:
#             j = i
#             remain = gas[i]-cost[i]
#             while True:
#                 j = (j+1)%n
#                 if j == i:
#                     return i
#                 remain += gas[j]-cost[j]
#                 if remain < 0:
#                     break
#     return -1


# if __name__ == '__main__':
#     gas = list(map(int, input("请输入gas：").split(',')))
#     cost = list(map(int, input("请输入cost：").split(',')))
#     print(canCompleteCircuit(gas,cost))


# 去除重复的字母：给你一个字符串s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

def removeDuplicateLetters(s:str)->str:
    stack = []
    for i in range(len(s)):
        if s[i] in stack:
            continue
        while stack and stack[-1] > s[i] and s.find(stack[-1],i) != -1: # 表示从i开始find
            stack.pop()
        stack.append(s[i])
    return ''.join(stack)

if __name__ == '__main__':
    s = input("请输入一个字符串：")
    print(removeDuplicateLetters(s))


