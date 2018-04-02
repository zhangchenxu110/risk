import pandas as pd
import numpy as np
import time
import datetime



# 函数定义
def count_order_by_1year(arr):
    if list(pd.notna(arr))[0]:
        _i = 0
        for x in iter(arr):
            if x is np.nan:
                return np.nan
            if x == 1:
                _i = _i + 1
        return _i
    else:
        return np.nan


def count_order_by_2year(arr):
    if list(pd.notna(arr))[0]:
        _i = 0
        for x in iter(arr):
            if x is np.nan:
                return np.nan
            if x == 2:
                _i = _i + 1
        return _i
    else:
        return np.nan


def count_order_by_3year(arr):
    if list(pd.notna(arr))[0]:
        _i = 0
        for x in iter(arr):
            if x is np.nan:
                return np.nan
            if x == 3:
                _i = _i + 1
        return _i
    else:
        return np.nan


def count_order_by_4year(arr):
    if list(pd.notna(arr))[0]:
        _i = 0
        for x in iter(arr):
            if x is np.nan:
                return np.nan
            if x >= 4:
                _i = _i + 1
        return _i
    else:
        return np.nan


# 距今年限
def year_before(x):
    if x is np.nan or x == 'nan':
        return x
    else:
        return 2018 - int(str(x).split('-')[0])


# id dataform 47031行 令行index是id 将其他表left join过来
df_list = pd.read_csv("E:\\risk\\test_list.csv", sep=',', skiprows=0, low_memory=False)
# 转成id appl_sbm_tm数据库 行index是id
id_unique_test = pd.DataFrame({"id": np.array(df_list['id'])}, index=np.array(df_list['id']))
# 将appl_sbm_tm转成距2017/6/1 天数
appl_sbm_tm_array = np.array(df_list['appl_sbm_tm'])
for i in range(len(appl_sbm_tm_array)):
    auth_arr = appl_sbm_tm_array[i].split(' ')[0].split('-')
    d1 = datetime.date(2017, 6, 1)
    d2 = datetime.date(int(auth_arr[0]), int(auth_arr[1]), int(auth_arr[2]))
    appl_sbm_tm_array[i] = (d1 - d2).days

id_unique_test['appl_sbm_tm'] = appl_sbm_tm_array
###############################################
# 一、order_info表 根据id groupby
###############################################
# id   申请贷款唯一编号
# no_order_md5 订单编号MD5加密
# name_rec_md5 收货人姓名MD5加密
# amt_order    订单金额
# type_pay 支付方式
# time_order   下单时间
# sts_order    订单状态
# phone    收货电话（脱敏）
# product_id_md5   商品编号MD5加密
# unit_price   商品单价
df_order_info = pd.read_csv("E:\\risk\\test_order_info.csv", sep=',', skiprows=0, low_memory=False)
# df_order_info.info() #737723行
# 0、先统计每个id的order的个数
order_countby_id = df_order_info["id"].groupby(df_order_info['id']).count()
id_unique_test = pd.concat([id_unique_test, order_countby_id], axis=1, join_axes=[id_unique_test.index])
id_unique_test.columns = ['id', 'appl_sbm_tm', 'order_count_id']  # 列重命名
# 1、根据id对amt_order金额总额进行groupby 之后赋到id_unique_test后面
amt_order_groupby_id = df_order_info['amt_order'].groupby(df_order_info['id']).mean()
id_unique_test = pd.concat([id_unique_test, amt_order_groupby_id], axis=1, join_axes=[id_unique_test.index])
# 2、phone转成每个id对应的个数 之后赋到id_unique_test后面
phone_count_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"phone": lambda x: x.nunique()})
id_unique_test = pd.concat([id_unique_test, phone_count_groupby_id], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'phone': 'phone_order_info', 'amt_order': 'amt_order_info'}, inplace=True)  # 列重命名
id_unique_test['phone_order_info'].replace(0, np.nan, inplace=True)
# 3、no_order_md5订单id 转成每个id对应的个数 之后赋到id_unique_test后面
no_order_md5_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"no_order_md5": lambda x: x.nunique()})
id_unique_test = pd.concat([id_unique_test, no_order_md5_groupby_id], axis=1, join_axes=[id_unique_test.index])
id_unique_test['no_order_md5'].replace(0, np.nan, inplace=True)
# 4、name_rec_md5收货人名字 转成去重每个id对应的个数  之后赋到id_unique_test后面
name_rec_md5_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"name_rec_md5": lambda x: x.nunique()})
id_unique_test = pd.concat([id_unique_test, name_rec_md5_groupby_id], axis=1, join_axes=[id_unique_test.index])
id_unique_test['name_rec_md5'].replace(0, np.nan, inplace=True)
# 5、type_pay 支付方式 分类型 按照种类分开计数 转成每个id对应的个数 存疑
type_pay_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"type_pay": lambda x: x.nunique()})
id_unique_test = pd.concat([id_unique_test, type_pay_groupby_id], axis=1, join_axes=[id_unique_test.index])
id_unique_test['type_pay'].replace(0, np.nan, inplace=True)
# 6、time_order 下单时间 按照年暂时分类 即2014及以前 2015 2016 2017年的下单数
# 先转换时间戳为时间
t_o = np.array(df_order_info['time_order'])
for i in range(len(t_o)):
    if len(str(t_o[i])) == 10:
        t_o[i] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(t_o[i])))

df_order_info["order_year_limit"] = list(map(year_before, t_o))
# 2014及以前 2015 2016 2017年的下单数 即距今4+  3 2 1
order_year4_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_4year})
order_year3_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_3year})
order_year2_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_2year})
order_year1_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_1year})
order_year4_limit_groupby_id.rename(columns={'order_year_limit': 'order_year4_limit_groupby_id'}, inplace=True)
order_year3_limit_groupby_id.rename(columns={'order_year_limit': 'order_year3_limit_groupby_id'}, inplace=True)
order_year2_limit_groupby_id.rename(columns={'order_year_limit': 'order_year2_limit_groupby_id'}, inplace=True)
order_year1_limit_groupby_id.rename(columns={'order_year_limit': 'order_year1_limit_groupby_id'}, inplace=True)
# 将结果拼入id_unique_test
id_unique_test = pd.concat(
    [id_unique_test, order_year4_limit_groupby_id, order_year3_limit_groupby_id, order_year2_limit_groupby_id,
     order_year1_limit_groupby_id], axis=1, join_axes=[id_unique_test.index])
# 7、sts_order 订单状态 分类型 按照种类分开计数 转成每个id对应的个数
# df_order_info['sts_order'].value_counts()
order_sts_list = ['完成', '充值成功', '已取消', '未抢中', '已完成', '订单取消', '等待收货', '出票成功',
                  '充值失败;退款成功', '退款完成', '等待付款', '充值失败', '出票失败', '已晒单', '正在出库',
                  '商品出库', '抢票已取消', '抢票已取消', '预订结束', '退款成功', '正在处理', '付款成功',
                  '失败退款', '失败退款', '等待审核', '等待处理', '已退款', '缴费成功', '配送退货',
                  '订单已取消', '请上门自提', '过期关闭', '等待退款', '预约完成', '未入住', '下单失败',
                  '已确认', '等待付款确认', '已入住', '正在充值', '商品退库', '已收货', '购买成功',
                  '正在送货（暂不能上门自提）', '过期放弃', '充值失败;退款处理中', '支付失败', '等待揭晓', '等待发码']
order_sts_list_val = [4, 4, 1, 2, 4, 1, 3, 3,
                      2, 2, 2, 2, 2, 5, 3,
                      3, 2, 2, 2, 2, 3, 4,
                      2, 2, 2, 2, 2, 4, 2,
                      2, 3, 1, 2, 2, 2, 2,
                      3, 2, 3, 3, 2, 4, 3,
                      3, 1, 2, 2, 3, 4]
df_order_info['sts_order'].replace(to_replace=order_sts_list, value=order_sts_list_val, inplace=True)
# df_order_info['sts_order'].fillna(value=0, inplace=True)
# df_order_info['sts_order'].astype(int)
sts_order_info = df_order_info["sts_order"].groupby(df_order_info['id']).mean()
id_unique_test = pd.concat([id_unique_test, sts_order_info], axis=1, join_axes=[id_unique_test.index])

# 8、product_id_md5 商品编号MD5加密  649819条 几乎全部   和no_order_md5订单id线性相关 暂时忽略这一行
# 9、unit_price 商品单价 目测NA较多 暂时不考虑

###############################################
# 二、bankcard_info表 96149行  银行卡 根据id groupby后47031行
################################################
#  将信用卡记录2分 储蓄卡记1分 并入id_unique_test
df_bankcard_info = pd.read_csv("E:\\risk\\test_bankcard_info.csv", sep=',', skiprows=0, low_memory=False)
df_bankcard_info['card_type'].replace(to_replace=['信用卡', '储蓄卡'], value=[2, 1], inplace=True)
card_type = df_bankcard_info["card_type"].groupby(df_bankcard_info['id']).mean()
id_unique_test = pd.concat([id_unique_test, card_type], axis=1, join_axes=[id_unique_test.index])
#   统计个数 phone 银行卡绑定手机号(脱敏)
phone_bankcard = df_bankcard_info["phone"].groupby(df_bankcard_info['id']).count()
id_unique_test = pd.concat([id_unique_test, phone_bankcard], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'phone': 'phone_bankcard'}, inplace=True)
id_unique_test['phone_bankcard'].replace(0, np.nan, inplace=True)
id_unique_test['card_type'].replace(0, np.nan, inplace=True)

# ###############################################
#  三、recieve_addr_info表 80382行  收货地址信息 根据id groupby后47031行
#  统计个数
# ###############################################
# addr_id 收货地址ID
df_recieve_addr_info = pd.read_csv("E:\\risk\\test_recieve_addr_info.csv", sep=',', skiprows=0, low_memory=False)
addr_id_recieve_addr = df_recieve_addr_info["addr_id"].groupby(df_recieve_addr_info['id']).count()
id_unique_test = pd.concat([id_unique_test, addr_id_recieve_addr], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'addr_id': 'addr_id_recieve_addr'}, inplace=True)
# region 收货地址所在地区
region_recieve_addr = df_recieve_addr_info["region"].groupby(df_recieve_addr_info['id']).count()
id_unique_test = pd.concat([id_unique_test, region_recieve_addr], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'region': 'region_recieve_addr'}, inplace=True)
# receiver_md5 收货人姓名(MD5加密)
receiver_recieve_addr = df_recieve_addr_info["receiver_md5"].groupby(df_recieve_addr_info['id']).count()
id_unique_test = pd.concat([id_unique_test, receiver_recieve_addr], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'receiver_md5': 'receiver_recieve_addr'}, inplace=True)
# phone 收货人手机号(脱敏)
phone_recieve_addr = df_recieve_addr_info["phone"].groupby(df_recieve_addr_info['id']).count()
id_unique_test = pd.concat([id_unique_test, phone_recieve_addr], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'phone': 'phone_recieve_addr'}, inplace=True)
# fix_phone 收货人固定电话号码(脱敏)
fix_phone_recieve_addr = df_recieve_addr_info["fix_phone"].groupby(df_recieve_addr_info['id']).count()
id_unique_test = pd.concat([id_unique_test, fix_phone_recieve_addr], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'phone': 'fix_phone_recieve_addr'}, inplace=True)
id_unique_test['addr_id_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_test['region_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_test['receiver_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_test['phone_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_test['fix_phone'].replace(0, np.nan, inplace=True)
###############################################
# 四、auth_info 认证信息表    47031
# id_card	身份证号（脱敏）  统计个数  即这个id身份证为Na就是0
# phone	认证电话号码（脱敏）  统计个数
# auth_time	认证时间
###############################################
df_auth_info = pd.read_csv("E:\\risk\\test_auth_info.csv", sep=',', skiprows=0, low_memory=False)
id_card_auth = df_auth_info["id_card"].groupby(df_auth_info['id']).count()
id_unique_test = pd.concat([id_unique_test, id_card_auth], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'id_card': 'id_card_auth'}, inplace=True)

phone_auth = df_auth_info["phone"].groupby(df_auth_info['id']).count()
id_unique_test = pd.concat([id_unique_test, phone_auth], axis=1, join_axes=[id_unique_test.index])
id_unique_test.rename(columns={'phone': 'phone_auth'}, inplace=True)
id_unique_test['phone_auth'].replace(0, np.nan, inplace=True)
id_unique_test['id_card_auth'].replace(0, np.nan, inplace=True)
# auth_time	认证时间 距离2018/3 相差月份
auth_time_array = np.array(df_auth_info['auth_time'])
for i in range(len(auth_time_array)):
    if auth_time_array[i] is not np.nan:
        auth_arr = auth_time_array[i].split('-')
        auth_year = 2018 - int(auth_arr[0])
        auth_mon = 3 - int(auth_arr[1])
        auth_time_array[i] = auth_year * 12 + auth_mon

df_auth_info['auth_time_before'] = auth_time_array
df_auth_info.index = df_auth_info['id']
id_unique_test = pd.concat([id_unique_test, df_auth_info['auth_time_before']], axis=1, join_axes=[id_unique_test.index])
###############################################
# 五、credit_info网络平台信用信息表     47031
# credit_score	网购平台信用评分
# quota	网购平台信用额度
# overdraft	网购平台信用额度使用值
###############################################
df_credit_info = pd.read_csv("E:\\risk\\test_credit_info.csv", sep=',', skiprows=0, low_memory=False)
df_credit_info.index = df_credit_info['id']
df_credit_info.drop('id', axis=1, inplace=True)
id_unique_test = pd.concat([id_unique_test, df_credit_info], axis=1, join_axes=[id_unique_test.index])
###############################################
# 六、userInfo表个人信息    47031
# sex	性别
# birthday	出生日期   转成年龄
# hobby	兴趣爱好
# merriage	婚姻状况
# income	收入水平
# id_card	身份证号(脱敏)  count 有身份证记1
# degree	学历
# industry	所在行业
# qq_bound	是否绑定QQ
# wechat_bound	是否绑定微信
# account_grade	会员级别
###############################################
df_user_info = pd.read_csv("E:\\risk\\test_user_info.csv", sep=',', skiprows=0, low_memory=False)
user_age_array = np.array(df_user_info['birthday'])
for i in range(len(user_age_array)):
    if user_age_array[i] is not np.nan:
        try:
            sp = user_age_array[i].split('-')
            if sp[0] == '90后':
                user_age_array[i] = 28
                continue
            if sp[0] == '80后':
                user_age_array[i] = 38
                continue
            user_age_array[i] = 2018 - int(sp[0]) if 100 > (2018 - int(sp[0])) > 0 else np.nan
        except Exception:
            user_age_array[i] = np.nan

df_user_info['user_age'] = user_age_array
df_user_info.index = df_user_info['id']
df_user_info.drop(labels=['id', 'birthday', 'id_card'], axis=1, inplace=True)
id_unique_test = pd.concat([id_unique_test, df_user_info], axis=1, join_axes=[id_unique_test.index])
# 写入csv
id_unique_test.to_csv('E:\\risk\\test.csv', na_rep='NA', index=False, sep=',')

#############################################################################################
# 测试集合并完毕
# 开始合并训练集
# 完成后将两个集合合并后进行预处理
#############################################################################################

# id dataform 47031行 令行index是id 将其他表left join过来
df_list = pd.read_csv("E:\\risk\\train_target.csv", sep=',', skiprows=0, low_memory=False)
# 转成id appl_sbm_tm数据库 行index是id
id_unique_train = pd.DataFrame({"id": np.array(df_list['id']), "target": np.array(df_list['target'])},
                               index=np.array(df_list['id']))
# 将appl_sbm_tm转成距2017/6/1 天数
appl_sbm_tm_array = np.array(df_list['appl_sbm_tm'])
for i in range(len(appl_sbm_tm_array)):
    auth_arr = appl_sbm_tm_array[i].split(' ')[0].split('-')
    d1 = datetime.date(2017, 6, 1)
    d2 = datetime.date(int(auth_arr[0]), int(auth_arr[1]), int(auth_arr[2]))
    appl_sbm_tm_array[i] = (d1 - d2).days

id_unique_train['appl_sbm_tm'] = appl_sbm_tm_array
###############################################
# 一、order_info表 根据id groupby
###############################################
# id   申请贷款唯一编号
# no_order_md5 订单编号MD5加密
# name_rec_md5 收货人姓名MD5加密
# amt_order    订单金额
# type_pay 支付方式
# time_order   下单时间
# sts_order    订单状态
# phone    收货电话（脱敏）
# product_id_md5   商品编号MD5加密
# unit_price   商品单价
df_order_info = pd.read_csv("E:\\risk\\train_order_info.csv", sep=',', skiprows=0, low_memory=False)
# df_order_info.info() #737723行
# 0、先统计每个id的order的个数
order_countby_id = df_order_info["id"].groupby(df_order_info['id']).count()
id_unique_train = pd.concat([id_unique_train, order_countby_id], axis=1, join_axes=[id_unique_train.index])
id_unique_train.columns = ['id', 'target', 'appl_sbm_tm', 'order_count_id']  # 列重命名
# 1、根据id对amt_order金额总额进行groupby 之后赋到id_unique_train后面
amt_order_groupby_id = df_order_info['amt_order'].groupby(df_order_info['id']).mean()
id_unique_train = pd.concat([id_unique_train, amt_order_groupby_id], axis=1, join_axes=[id_unique_train.index])
# 2、phone转成每个id对应的个数 之后赋到id_unique_train后面
phone_count_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"phone": lambda x: x.nunique()})
id_unique_train = pd.concat([id_unique_train, phone_count_groupby_id], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'phone': 'phone_order_info', 'amt_order': 'amt_order_info'}, inplace=True)  # 列重命名
id_unique_train['phone_order_info'].replace(0, np.nan, inplace=True)
# 3、no_order_md5订单id 转成每个id对应的个数 之后赋到id_unique_train后面
no_order_md5_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"no_order_md5": lambda x: x.nunique()})
id_unique_train = pd.concat([id_unique_train, no_order_md5_groupby_id], axis=1, join_axes=[id_unique_train.index])
id_unique_train['no_order_md5'].replace(0, np.nan, inplace=True)
# 4、name_rec_md5收货人名字 转成去重每个id对应的个数  之后赋到id_unique_train后面
name_rec_md5_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"name_rec_md5": lambda x: x.nunique()})
id_unique_train = pd.concat([id_unique_train, name_rec_md5_groupby_id], axis=1, join_axes=[id_unique_train.index])
id_unique_train['name_rec_md5'].replace(0, np.nan, inplace=True)
# 5、type_pay 支付方式 分类型 按照种类分开计数 转成每个id对应的个数 存疑
type_pay_groupby_id = df_order_info.groupby(df_order_info['id']).agg({"type_pay": lambda x: x.nunique()})
id_unique_train = pd.concat([id_unique_train, type_pay_groupby_id], axis=1, join_axes=[id_unique_train.index])
id_unique_train['type_pay'].replace(0, np.nan, inplace=True)
# 6、time_order 下单时间 按照年暂时分类 即2014及以前 2015 2016 2017年的下单数
# 先转换时间戳为时间
t_o = np.array(df_order_info['time_order'])
for i in range(len(t_o)):
    if len(str(t_o[i])) == 10:
        t_o[i] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(t_o[i])))

df_order_info["order_year_limit"] = list(map(year_before, t_o))
# 2014及以前 2015 2016 2017年的下单数 即距今4+  3 2 1
order_year4_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_4year})
order_year3_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_3year})
order_year2_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_2year})
order_year1_limit_groupby_id = df_order_info.groupby(df_order_info['id']).agg(
    {"order_year_limit": count_order_by_1year})
order_year4_limit_groupby_id.rename(columns={'order_year_limit': 'order_year4_limit_groupby_id'}, inplace=True)
order_year3_limit_groupby_id.rename(columns={'order_year_limit': 'order_year3_limit_groupby_id'}, inplace=True)
order_year2_limit_groupby_id.rename(columns={'order_year_limit': 'order_year2_limit_groupby_id'}, inplace=True)
order_year1_limit_groupby_id.rename(columns={'order_year_limit': 'order_year1_limit_groupby_id'}, inplace=True)
# 将结果拼入id_unique_train
id_unique_train = pd.concat(
    [id_unique_train, order_year4_limit_groupby_id, order_year3_limit_groupby_id, order_year2_limit_groupby_id,
     order_year1_limit_groupby_id], axis=1, join_axes=[id_unique_train.index])
# 7、sts_order 订单状态 分类型 按照种类分开计数 转成每个id对应的个数
# df_order_info['sts_order'].value_counts()
order_sts_list = ['完成', '充值成功', '已取消', '未抢中', '已完成', '订单取消', '等待收货', '出票成功',
                  '充值失败;退款成功', '退款完成', '等待付款', '充值失败', '出票失败', '已晒单', '正在出库',
                  '商品出库', '抢票已取消', '抢票已取消', '预订结束', '退款成功', '正在处理', '付款成功',
                  '失败退款', '失败退款', '等待审核', '等待处理', '已退款', '缴费成功', '配送退货',
                  '订单已取消', '请上门自提', '过期关闭', '等待退款', '预约完成', '未入住', '下单失败',
                  '已确认', '等待付款确认', '已入住', '正在充值', '商品退库', '已收货', '购买成功',
                  '正在送货（暂不能上门自提）', '过期放弃', '充值失败;退款处理中', '支付失败', '等待揭晓', '等待发码',
                  '发货中', '预订中', '部分充值成功;退款成功', '退款中', '已取消订单', '等待分期付款', '等待厂商处理']
order_sts_list_val = [4, 4, 1, 2, 4, 1, 3, 3,
                      2, 2, 2, 2, 2, 5, 3,
                      3, 2, 2, 2, 2, 3, 4,
                      2, 2, 2, 2, 2, 4, 2,
                      2, 3, 1, 2, 2, 2, 2,
                      3, 2, 3, 3, 2, 4, 3,
                      3, 1, 2, 2, 3, 4,
                      3, 2, 3, 1, 1, 2, 2]
df_order_info['sts_order'].replace(to_replace=order_sts_list, value=order_sts_list_val, inplace=True)
# df_order_info['sts_order'].fillna(value=0, inplace=True)
# df_order_info['sts_order'].astype(int)
sts_order_info = df_order_info["sts_order"].groupby(df_order_info['id']).mean()
id_unique_train = pd.concat([id_unique_train, sts_order_info], axis=1, join_axes=[id_unique_train.index])

# for d in df_order_info["sts_order"]:
#     if type(d) is not int:
#         if pd.notna(d):
#             d

# 8、product_id_md5 商品编号MD5加密  649819条 几乎全部   和no_order_md5订单id线性相关 暂时忽略这一行
# 9、unit_price 商品单价 目测NA较多 暂时不考虑

###############################################
# 二、bankcard_info表 96149行  银行卡 根据id groupby后47031行
################################################
#  将信用卡记录2分 储蓄卡记1分 并入id_unique_train
df_bankcard_info = pd.read_csv("E:\\risk\\train_bankcard_info.csv", sep=',', skiprows=0, low_memory=False)
df_bankcard_info['card_type'].replace(to_replace=['信用卡', '储蓄卡'], value=[2, 1], inplace=True)
card_type = df_bankcard_info["card_type"].groupby(df_bankcard_info['id']).mean()
id_unique_train = pd.concat([id_unique_train, card_type], axis=1, join_axes=[id_unique_train.index])
#   统计个数 phone 银行卡绑定手机号(脱敏)
phone_bankcard = df_bankcard_info["phone"].groupby(df_bankcard_info['id']).count()
id_unique_train = pd.concat([id_unique_train, phone_bankcard], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'phone': 'phone_bankcard'}, inplace=True)
id_unique_train['phone_bankcard'].replace(0, np.nan, inplace=True)
id_unique_train['card_type'].replace(0, np.nan, inplace=True)

# ###############################################
#  三、recieve_addr_info表 80382行  收货地址信息 根据id groupby后47031行
#  统计个数
# ###############################################
# addr_id 收货地址ID
df_recieve_addr_info = pd.read_csv("E:\\risk\\train_recieve_addr_info.csv", sep=',', skiprows=0, low_memory=False)
addr_id_recieve_addr = df_recieve_addr_info["addr_id"].groupby(df_recieve_addr_info['id']).count()
id_unique_train = pd.concat([id_unique_train, addr_id_recieve_addr], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'addr_id': 'addr_id_recieve_addr'}, inplace=True)
# region 收货地址所在地区
region_recieve_addr = df_recieve_addr_info["region"].groupby(df_recieve_addr_info['id']).count()
id_unique_train = pd.concat([id_unique_train, region_recieve_addr], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'region': 'region_recieve_addr'}, inplace=True)
# receiver_md5 收货人姓名(MD5加密)
receiver_recieve_addr = df_recieve_addr_info["receiver_md5"].groupby(df_recieve_addr_info['id']).count()
id_unique_train = pd.concat([id_unique_train, receiver_recieve_addr], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'receiver_md5': 'receiver_recieve_addr'}, inplace=True)
# phone 收货人手机号(脱敏)
phone_recieve_addr = df_recieve_addr_info["phone"].groupby(df_recieve_addr_info['id']).count()
id_unique_train = pd.concat([id_unique_train, phone_recieve_addr], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'phone': 'phone_recieve_addr'}, inplace=True)
# fix_phone 收货人固定电话号码(脱敏)
fix_phone_recieve_addr = df_recieve_addr_info["fix_phone"].groupby(df_recieve_addr_info['id']).count()
id_unique_train = pd.concat([id_unique_train, fix_phone_recieve_addr], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'phone': 'fix_phone_recieve_addr'}, inplace=True)
id_unique_train['addr_id_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_train['region_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_train['receiver_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_train['phone_recieve_addr'].replace(0, np.nan, inplace=True)
id_unique_train['fix_phone'].replace(0, np.nan, inplace=True)
###############################################
# 四、auth_info 认证信息表    47031
# id_card	身份证号（脱敏）  统计个数  即这个id身份证为Na就是0
# phone	认证电话号码（脱敏）  统计个数
# auth_time	认证时间
###############################################
df_auth_info = pd.read_csv("E:\\risk\\train_auth_info.csv", sep=',', skiprows=0, low_memory=False)
id_card_auth = df_auth_info["id_card"].groupby(df_auth_info['id']).count()
id_unique_train = pd.concat([id_unique_train, id_card_auth], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'id_card': 'id_card_auth'}, inplace=True)

phone_auth = df_auth_info["phone"].groupby(df_auth_info['id']).count()
id_unique_train = pd.concat([id_unique_train, phone_auth], axis=1, join_axes=[id_unique_train.index])
id_unique_train.rename(columns={'phone': 'phone_auth'}, inplace=True)
id_unique_train['phone_auth'].replace(0, np.nan, inplace=True)
id_unique_train['id_card_auth'].replace(0, np.nan, inplace=True)
# auth_time	认证时间 距离2018/3 相差月份
auth_time_array = np.array(df_auth_info['auth_time'])
for i in range(len(auth_time_array)):
    if auth_time_array[i] is not np.nan:
        auth_arr = auth_time_array[i].split('-')
        auth_year = 2018 - int(auth_arr[0])
        auth_mon = 3 - int(auth_arr[1])
        auth_time_array[i] = auth_year * 12 + auth_mon

df_auth_info['auth_time_before'] = auth_time_array
df_auth_info.index = df_auth_info['id']
id_unique_train = pd.concat([id_unique_train, df_auth_info['auth_time_before']], axis=1,
                            join_axes=[id_unique_train.index])
###############################################
# 五、credit_info网络平台信用信息表     47031
# credit_score	网购平台信用评分
# quota	网购平台信用额度
# overdraft	网购平台信用额度使用值
###############################################
df_credit_info = pd.read_csv("E:\\risk\\train_credit_info.csv", sep=',', skiprows=0, low_memory=False)
df_credit_info.index = df_credit_info['id']
df_credit_info.drop('id', axis=1, inplace=True)
id_unique_train = pd.concat([id_unique_train, df_credit_info], axis=1, join_axes=[id_unique_train.index])
###############################################
# 六、userInfo表个人信息    47031
# sex	性别
# birthday	出生日期   转成年龄
# hobby	兴趣爱好
# merriage	婚姻状况
# income	收入水平
# id_card	身份证号(脱敏)  count 有身份证记1
# degree	学历
# industry	所在行业
# qq_bound	是否绑定QQ
# wechat_bound	是否绑定微信
# account_grade	会员级别
###############################################
df_user_info = pd.read_csv("E:\\risk\\train_user_info.csv", sep=',', skiprows=0, low_memory=False)
user_age_array = np.array(df_user_info['birthday'])
for i in range(len(user_age_array)):
    if user_age_array[i] is not np.nan:
        try:
            sp = user_age_array[i].split('-')
            if sp[0] == '90后':
                user_age_array[i] = 28
                continue
            if sp[0] == '80后':
                user_age_array[i] = 38
                continue
            user_age_array[i] = 2018 - int(sp[0]) if 100 > (2018 - int(sp[0])) > 0 else np.nan
        except Exception:
            user_age_array[i] = np.nan

df_user_info['user_age'] = user_age_array
df_user_info.index = df_user_info['id']
df_user_info.drop(labels=['id', 'birthday', 'id_card'], axis=1, inplace=True)
id_unique_train = pd.concat([id_unique_train, df_user_info], axis=1, join_axes=[id_unique_train.index])

for index, row in id_unique_train.iterrows():  # 获取每行的index、row 删除缺失值大于80%的行
    if (1 - row.count() / len(row)) > 0.8:
        id_unique_train.drop(index, axis=0, inplace=True)
# 写入csv
id_unique_train.to_csv('E:\\risk\\train.csv', na_rep='NA', index=False, sep=',')
