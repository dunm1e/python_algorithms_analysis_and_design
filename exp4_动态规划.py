from typing import List


# 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。

# def max_cube(maxtrix: List[List[int]], m: int, n: int) -> int:
#     dp = [[0 for i in range(n)] for j in range(m)]
#     max_side = 0
#     for i in range(m):
#         for j in range(n):
#             if i == 0 or j == 0:
#                 dp[i][j] = maxtrix[i][j]  # 先把第一行和第一列的值赋值给dp
#             else:
#                 if maxtrix[i][j] == 1:
#                     dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j],
#                                    dp[i][j - 1]) + 1  # 表示如果左上、上、左均为1，那么这个方格的值就是2，也就是是一个2*2的正方形
#             max_side = max(max_side, dp[i][j])  # 每次都要更新最大边长
#     return max_side * max_side


# if __name__ == '__main__':
#     m, n = map(int, input("请输入行 列：").split())
#     matrix = []
#     for i in range(m):
#         row = list(map(int, input(f"请输入第{i + 1}行：").split()))
#         matrix.append(row)
#     print("最大正方形面积为：" + str(max_cube(matrix, m, n)))


# 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。
# def numSquares(n: int) -> int:
#     dp = [float('inf')] * (n + 1)
#     dp[0] = 0
#     dp[1] = 1
#     for i in range(2, n + 1):
#         j = 1
#         while j * j <= i:
#             dp[i] = min(dp[i - j * j] + 1, dp[i])
#             j += 1
#     return dp[n]


# if __name__ == '__main__':
#     n = int(input("请输入一个整数："))
#     print("完全平方数的最少数量为：" + str(numSquares(n)))


# 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。
# 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
# 请你计算并返回达到楼梯顶部的最低花费。
# def minCostClimbingStairs(cost: List[int]) -> int:
#     n = len(cost)
#     if n == 2:
#         return min(cost[0], cost[1])
#     if n < 2:
#         return 0
#     dp = [0] * (n + 1)
#     dp[0] = cost[0]
#     dp[1] = cost[1]
#     for i in range(2, n):
#         dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
#     return min(dp[n - 1], dp[n - 2])


# if __name__ == '__main__':
#     cost = list(map(int, input("请输入一个整数数组：").split(',')))
#     print("最低花费为：" + str(minCostClimbingStairs(cost)))


# 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
# 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
# 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
# 注意：不允许旋转信封。


def maxEnvelopes(envelopes: List[List[int]]) -> int:
    n = len(envelopes)
    if n == 0:
        return 0
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    print(envelopes)
    height = [envelopes[i][1] for i in range(n)]
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if height[i] > height[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


if __name__ == "__main__":
    envelopes = []
    n = int(input("请输入信封个数："))
    for i in range(n):
        row = list(map(int, input(f"请输入第{i + 1}个信封的宽度和高度：").split()))
        envelopes.append(row)
    print(maxEnvelopes(envelopes))

    # 最长上升子序列：子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
    # 例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
    # class Solution:
    #     def lengthOfLIS(self, nums: List[int]) -> int:
    #         if not nums:
    #             return 0
    #         dp = [1] * len(nums)
    #         for i in range(len(nums)):
    #             for j in range(i):
    #                 if nums[j] < nums[i]:
    #                     dp[i] = max(dp[i], dp[j] + 1)
    #         return max(dp)

    # if __name__ == '__main__':
    #     nums = list(map(int, input("请输入一个整数数组：").split(',')))
    #     solution = Solution()
    #     print("最长上升子序列长度为：" + str(solution.lengthOfLIS(nums)))


# 粉刷房子：假如有一排房子，共 n 个，每个房子可以被粉刷成红色、蓝色或者绿色这三种颜色中的一种，你
# 需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。当然，因为市场上不同颜色油漆的价格不同，
# 所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 n x 3 的正整数
# 矩阵 costs 来表示的。例如，costs[0][0] 表示第 0 号房子粉刷成红色的成本花费；costs[1][2] 表示第 1 号房子
# 粉刷成绿色的花费，以此类推。请计算出粉刷完所有房子最少的花费成本
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        dp = costs[0]
        for i in range(1, len(costs)):
            dp = [min(dp[j - 1], dp[j - 2]) + c for j, c in enumerate(costs[i])]
        return min(dp)
