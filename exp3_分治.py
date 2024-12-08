from typing import List


# 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
# 子数组是数组中的一个连续部分。
# 动态规划
# def maxsum(nums: List[int]) -> int:
#     len_arr = len(nums)
#     if len_arr == 1:
#         return nums[0]
#     sum = nums[0]
#     max = nums[0]
#     for i in range(1, len_arr):
#         if sum < 0:
#             sum = nums[i]
#         else:
#             sum += nums[i]
#         if sum > max:
#             max = sum
#     return max


# if __name__ == '__main__':
#     num = input("enter a array:")
#     a = num.split(',')
#     nums = []
#     for i in a:
#         nums.append(int(i))
#     print(maxsum(nums))

#分治法
# def maxsum(nums: List[int]) -> int:
#     len_arr = len(nums)
#     if len_arr == 1:
#         return nums[0]
#     else:
#         left = 0
#         right = len_arr - 1
#         return maxsum_divide(nums, left, right)


# def maxsum_divide(nums: List[int], left: int, right: int) -> int:
#     if left == right:
#         return nums[left]
#     mid = (left + right) // 2
#     left_max = maxsum_divide(nums, left, mid)
#     right_max = maxsum_divide(nums, mid + 1, right)
#     left_sum = nums[mid]
#     left_max_sum = nums[mid]
#     for i in range(mid - 1, left - 1, -1):
#         left_sum += nums[i]
#         if left_sum > left_max_sum:
#             left_max_sum = left_sum
#     right_sum = nums[mid + 1]
#     right_max_sum = nums[mid + 1]
#     for i in range(mid + 2, right + 1):
#         right_sum += nums[i]
#         if right_sum > right_max_sum:
#             right_max_sum = right_sum
#     cross_max = left_max_sum + right_max_sum
#     return max(left_max, right_max, cross_max)


# if __name__ == '__main__':
#     num = input("enter a array:")
#     a = num.split(',')
#     nums = []
#     for i in a:
#         nums.append(int(i))
#     print(maxsum(nums))

# 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
# 你可以假设数组是非空的，并且给定的数组总是存在多数元素。

# def most_num1(nums: List[int]) -> int:
#     nums.sort()
#     return nums[len(nums) // 2]


# def most_num2(nums: List[int]) -> int:
#     nums.sort()
#     j = k = 0
#     count = 0
#     for i in range(len(nums)):
#         if nums[k] == nums[j]:
#             k += 1
#             count += 1
#             if count > (len(nums) // 2):
#                 return nums[j]
#         else:
#             j = k
#             k += 1
#             count = 1


# if __name__ == '__main__':
#     num = input("enter a array:")
#     a = num.split(',')
#     nums = []
#     for i in a:
#         nums.append(int(i))
#     print(most_num2(nums))

# 实现 pow(x, n) ，即计算 x 的整数 n 次幂函数（即，x^𝑛 ）。
def pow_self(x: float, n: int) -> float:
    if x == 0:
        return 0
    if n == 0:
        return 1
    elif n % 2 == 0:
        half_n = n // 2
        half_result = pow_self(x, half_n)
        return half_result * half_result
    else:
        half_n = (n - 1) // 2
        half_result = pow_self(x, half_n)
        return half_result * half_result * x


if __name__ == '__main__':
    x = float(input("enter a base:"))
    n = int(input("enter a exponent:"))
    print('%.5f'% pow_self(x, n))
