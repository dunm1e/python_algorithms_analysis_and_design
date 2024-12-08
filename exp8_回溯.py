from typing import List

# # 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
# # 示例: 输入: n = 4, k = 2 输出: [ [2,4], [3,4], [2,3], [1,2], [1,3], [1,4]]

# def backtrack(start:int,track:List[int],n:int,k:int,res:List[List[int]]):
#     if len(track) == k:
#         res.append(track[:])
#         return
#     for i in range(start,n+1):
#         track.append(i)
#         backtrack(i+1,track,n,k,res)
#         track.pop()

# def combine(n:int,k:int)->List[List[int]]:
#     res = []
#     track = []
#     backtrack(1,track,n,k,res)
#     return res

# if __name__ == '__main__':
#     n = int(input("请输入n："))
#     k = int(input("请输入k："))
#     print(combine(n,k))

# 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
# 说明：所有数字都是正整数。解集不能包含重复的组合。
# 示例 1: 输入: k = 3, n = 7 输出: [[1,2,4]]
# 示例 2: 输入: k = 3, n = 9 输出: [[1,2,6], [1,3,5], [2,3,4]]

# def backtrack(start:int,track:List[int],n:int,k:int,res:List[List[int]]):
#     if sum(track) > n:
#         return
#     if len(track) == k and sum(track) == n:
#         res.append(track[:])
#         return
#     for i in range(start,10):
#         track.append(i)
#         backtrack(i+1,track,n,k,res)
#         track.pop()

# def CombinationSum(k:int,n:int)->List[List[int]]:
#     res = []
#     track = []
#     backtrack(1,track,n,k,res)
#     return res


# if __name__ == '__main__':
#     k = int(input("请输入k："))
#     n = int(input("请输入n："))
#     print(CombinationSum(k,n))

# 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
# 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
# 示例: 输入："23" 输出: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

def backtrack(start:int,track:str,digits:str,res:List[str],phone:List[str]):
    if len(track)==len(digits):
        res.append(track)
        return
    for i in range(start,len(digits)):
        for j in phone[int(digits[i])-2]:
            track+=j
            backtrack(i+1,track,digits,res,phone)
            track = track[:-1]

def LetterCombinations(digits:str)->List[str]:
    res = []
    phone = ["abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]
    track = ""
    backtrack(0,track,digits,res,phone)
    return res


if __name__ == '__main__':
    digits = input("请输入数字：")
    print(LetterCombinations(digits))