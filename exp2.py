## 方法一
# a = input("enter a array:")
# nums = []
# num = a.split(',')
# for i in num:
#     nums.append(int(i))
# target = int(input("enter a target:"))
#
# low = 0
# high = len(nums) - 1
# x = y = -1
# # 找出target在nums中的开始位置和结束位置，时间复杂度为O(logn)
# while low <= high:
#     mid = low + (high - low) >> 2
#     if nums[mid] < target:
#         low = mid + 1
#     elif nums[mid] > target:
#         high = mid - 1
#     else:
#         x = y = mid
#         while x > 0 and nums[x] == nums[x - 1]:
#             x -= 1
#         while y < len(nums) - 1 and nums[y] == nums[y + 1]:
#             y += 1
#         break
# print([x, y])


# lower_bound 返回最小的满足 nums[i] >= target 的 i
# 如果数组为空，或者所有数都 < target，则返回 len(nums)
# 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]

def lower_bound(nums, target):
    left, right = 0, len(nums) - 1  # 在区间 [left, right] 寻找解
    while left <= right:  # 区间不为空
        # 循环不变量：
        # nums[left-1] < target
        # nums[right+1] >= target
        mid = left + ((right - left) >> 1)
        if nums[mid] < target:  # 必须让left始终小于target，若中间值等于target，改变的是right
            left = mid + 1  # 范围缩小到 [mid+1, right]
        else:
            right = mid - 1  # 范围缩小到 [left, mid-1]
    return left

class Solution:
    def searchRange(self, nums, target):
        start = lower_bound(nums, target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        # 如果 start 存在，那么 end 必定存在
        end = lower_bound(nums, target + 1) - 1  #找到比target大1的数的位置再减去1就得到了right
        return [start, end]


if __name__ == '__main__':
    a = input("enter a array:")
    nums = []
    num = a.split(',')
    for i in num:
        nums.append(int(i))
    target = int(input("enter a target:"))
    print(Solution().searchRange(nums, target))
