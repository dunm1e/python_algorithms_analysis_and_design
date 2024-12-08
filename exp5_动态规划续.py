from typing import List


# 乘积最大子数组：给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组
# 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
# def maxProduct(nums: List[int]) -> int:
#     n = len(nums)
#     if n == 0:
#         return 0
#     if n == 1:
#         return nums[0]
#     dp = [[0] * 2 for _ in range(n)]
#     dp[0][0] = nums[0]
#     dp[0][1] = nums[0]
#     res = nums[0]
#     for i in range(1, n):
#         dp[i][0] = max(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i], nums[i])
#         dp[i][1] = min(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i], nums[i]) # 由于存在负数，所以还需要维护一个最小值
#         res = max(res, dp[i][0])
#     return res


# if __name__ == "__main__":
#     nums = list(map(int, input("请输入一个整数数组：").split(',')))
#     print("乘积最大子数组为：" + str(maxProduct(nums)))

# 买卖股票的最佳时机：给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。你只
# 能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能
# 获取的最大利润。返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

# def maxProfit(prices: List[int]) -> int:
#     n = len(prices)
#     if n == 0:
#         return 0
#     dp = [[0] * 2 for _ in range(n)]
#     dp[0][0] = 0
#     dp[0][1] = -prices[0]
#     for i in range(1, n):
#         dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
#         dp[i][1] = max(dp[i - 1][1], -prices[i])
#     return dp[n - 1][0]


# if __name__ == "__main__":
#     prices = list(map(int, input("请输入一个价格数组：").split(',')))
#     print("最大利润为：" + str(maxProfit(prices)))

# 奇怪的打印机：有台奇怪的打印机有以下两个特殊要求：1）打印机每次只能打印由 同一个字符 组成的序
# 列。2）每次可以在从起始到结束的任意位置打印新字符，并且会覆盖掉原来已有的字符。
# 给你一个字符串 s ，你的任务是计算这个打印机打印它需要的最少打印次数。
# def strangePrinter(s: str) -> int:
#     n = len(s)
#     dp = [[0] * n for _ in range(n)]
#     for i in range(n - 1, -1, -1):
#         dp[i][i] = 1
#         for j in range(i + 1, n):
#             if s[i] == s[j]: # 如果两个字符相等，那么打印i到j的字符的次数和打印i到j-1的字符的次数相同
#                 dp[i][j] = dp[i][j - 1]
#             else:
#                 minn = float('inf') # 如果两个字符不相等，那么打印i到j的字符的次数等于打印i到k的字符的次数加上打印k+1到j的字符的次数
#                 for k in range(i, j):
#                     minn = min(minn, dp[i][k] + dp[k + 1][j])
#                 dp[i][j] = minn
#     return dp[0][n - 1] # 返回打印0到n-1的字符的次数


# if __name__ == "__main__":
#     s = input("请输入一个字符串：")
#     print("最少打印次数为：" + str(strangePrinter(s)))

# 你会得到一个字符串 text 。你应该把它分成 k 个子字符串 (subtext1, subtext2，…， subtextk) ，要求满足:subtext_i 是 非空字符串
# 所有子字符串的连接等于 text ( 即subtext_1 + subtext_2 + ... + subtext_k == text )
# 对于所有 i 的有效值( 即 1 <= i <= k ) ，subtext_i == subtext_(k - i + 1) 均成立
# 返回k可能最大值

# 动态规划
def longestDecomposition(text: str) -> int:
        n = len(text)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        l, r = (n - 1) >> 1, n >> 1
        while l >= 0:
            dp[l][r] = max(dp[l][r], 1)
            if l != r:
                for i in range(l + 1):
                    if text[i:l + 1] == text[r:r + l + 1 - i]:
                        dp[i][r + l - i] = max(dp[i][r + l - i], dp[l + 1][r - 1] + 2)
            l -= 1
            r += 1

        return dp[0][n - 1]
if __name__ =="__main__":
    text = input("请输入一个字符串：")
    print(longestDecomposition(text))

# 贪心
# def longestDecomposition(text: str) -> int:
#     for i in range(1, (len(text) >> 1) + 1):  # 将原字符串分成两半，可以理解为双指针，因为如果有相同字符串不可能超过原来的一半长
#         if text[:i] == text[-i:]:
#             return longestDecomposition(text[i:-i]) + 2  # 找到相同的字符串直接返回进行递归
#     return min(1, len(text))  # 如果没有相同的，直接返回1，就是剩下的字符合起来为一个字符串
#
#
