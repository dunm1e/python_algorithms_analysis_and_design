# # 找丑数
# def isugly(num):
#     while(num%2==0):
#         num = num/2
#     while(num%3==0):
#         num = num/3
#     while(num%5==0):
#         num = num/5
#     if(num==1):
#         return True
#     else:
#          return False
# if __name__ == '__main__':
#     n = int(input("请输入一个整数："))
#     count = 0
#     num = 1
#     while(count!=n):
#         if(isugly(num)):
#             count+=1
#             num+=1
#         else:
#             num+=1
#     print("第",n,"个丑数是：",num-1)


# # 栅栏涂色
# def fence(n, k):
#     if n == 1:
#         return k
#     if n == 2:
#         return k * k
#     dp = [0] * (n + 1)
#     dp[1] = k
#     dp[2] = k * k
#     for i in range(3, n + 1):
#         dp[i] = (dp[i - 1] + dp[i - 2]) * (k - 1)
#     return dp[n]


# if __name__ == '__main__':
#     n = int(input("n = "))
#     k = int(input("k = "))
#     print(fence(n, k))


# # 会议室数量
# def meeting_room(intervals):
#     if not intervals:
#         return 0
#     intervals.sort(key=lambda x:x[0])
#     if(len(intervals)==1):
#         return 1
#     count = 1
#     for i in range(1,len(intervals)):
#         if intervals[i][0]<intervals[i-1][1]:
#             count+=1
#     return count


# if __name__ == '__main__':
#     intervals = [[0,30],[5,10],[15,20]]
#     print(meeting_room(intervals))



# 字母间隔
def char_interval(s,k):
    chars = [0]*26
    res = ['0']*len(s)
    for i in range(len(s)):
        chars[ord(s[i])-ord('a')]+=1
    for i in range(26):
        while chars[i]!=0:
            for j in range(i,len(s),k):
                if res[j]=='0':
                    res[j] = chr(i+ord('a'))
                    chars[i]-=1
                    break
            #如果无法间隔，返回空字符串
            else:
                return ''
    return ''.join(res)


if __name__ == '__main__':
    s = input("请输入一个字符串：")
    k = int(input("请输入一个整数："))
    print(char_interval(s,k))




# # 构造最近时间
# def closest_time(time):
#     a = int(time[0])
#     b = int(time[1])
#     c = int(time[3])
#     d = int(time[4])
#     now_time = a*600 + b*60 + c*10 + d
#     nums = [a, b, c, d]
#     min_diff = 24*60
#     for i in range(4):
#         for j in range(4):
#             for k in range(4):
#                 for l in range(4):
#                     if nums[i]*10+nums[j] < 24 and nums[k]*10+nums[l] < 60:
#                         new_time = nums[i]*600 + nums[j]*60 + nums[k]*10 + nums[l]
                    
#                         if new_time == now_time:
#                             continue
#                         elif new_time > now_time:
#                             next = new_time - now_time 
                            
#                         elif new_time < now_time:
#                             next = 24*60 - now_time + new_time
#                         if next < min_diff:
#                             min_diff = next
#                             res = str(nums[i])+str(nums[j])+':'+str(nums[k])+str(nums[l])
#     return res


# if __name__ == '__main__':
#     time = input("请输入一个时间：")
#     print(closest_time(time))
                                
