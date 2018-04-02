import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import datetime


# 先将训练集合测试集合并做预处理
df_train = pd.read_csv("E:\\risk\\train.csv", sep=',', skiprows=0, low_memory=False)
df_test = pd.read_csv("E:\\risk\\test.csv", sep=',', skiprows=0, low_memory=False)
feature = df_train['target']
id_train = df_train['id']
id_test = df_test['id']
df_train.drop(['id', 'target'], axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)
df_train_row = df_train.shape[0]
df_all = pd.concat([df_train, df_test])
# 检查列的缺失值
df_all.assign(miss=df_all.apply(lambda x: (len(x) - x.count()) / float(len(x))))

# account_grade                   0.464367  补充‘未知’
# addr_id_recieve_addr            0.538503  补充-10
# amt_order_info                  0.475345  补充均值 增加一列 isna
# appl_sbm_tm                     0.000000
# auth_time_before                0.349631  补充-50
# card_type                       0.000000
# credit_score                    0.014539  补充0
# degree                          0.937050  删了
# fix_phone                       0.586497  补充-5
# hobby                           0.883621  删了
# id                              0.000000
# id_card_auth                    0.323131  补充0
# income                          0.930549  删了
# industry                        0.937717  删了
# merriage                        0.932954  删了
# name_rec_md5                    0.478054  补充-5
# no_order_md5                    0.474976  补充-100
# order_count_id                  0.000000
# order_year1_limit_groupby_id    0.474976  补充0
# order_year2_limit_groupby_id    0.474976  补充0
# order_year3_limit_groupby_id    0.474976  补充0
# order_year4_limit_groupby_id    0.474976  补充0 增加一列 isna
# overdraft                       0.014539  补充0 增加一列 isna
# phone_auth                      0.025845  补充0
# phone_bankcard                  0.000000
# phone_order_info                0.477233  补充-10
# phone_recieve_addr              0.297231  补充-10
# qq_bound                        0.037759  补充‘未知’
# quota                           0.014539  补充0 增加一列 isna
# receiver_recieve_addr           0.297237  补充-10
# region_recieve_addr             0.297231  补充-10
# sex                             0.030192  补充‘未知’
# sts_order                       0.501727  补充-10
# target                          0.280013
# type_pay                        0.478019  补充-5
# user_age                        0.667814  补充-100
# wechat_bound                    0.037759  补充‘未知’

# account_grade                   0.464367  补充‘未知’
df_all['account_grade'].value_counts()  # 五类
df_all['account_grade'].fillna('未知', inplace=True)
# addr_id_recieve_addr            0.538503  补充-10
df_all['addr_id_recieve_addr'].fillna(-10, inplace=True)
# amt_order_info                  0.475345  补充均值 增加一列 isna
df_all.loc[(df_all.amt_order_info.notnull()), 'amt_order_info_isna'] = "0"
df_all.loc[(df_all.amt_order_info.isnull()), 'amt_order_info_isna'] = "1"
df_all['amt_order_info'].interpolate(inplace=True)  # 插值法
# appl_sbm_tm                     0.000000
# auth_time_before                0.349631  补充-50
df_all['auth_time_before'].fillna(-50, inplace=True)
# card_type                       0.000000
# credit_score                    0.014539  补充0
df_all['credit_score'].fillna(0, inplace=True)
# degree                          0.937050  删了
df_all.drop('degree', axis=1, inplace=True)
# fix_phone                       0.586497  补充-5
df_all['fix_phone'].fillna(-5, inplace=True)
# hobby                           0.883621  删了
df_all.drop('hobby', axis=1, inplace=True)
# id                              0.000000
# id_card_auth                    0.323131  补充0
df_all['id_card_auth'].fillna(0, inplace=True)
# income                          0.930549  删了
df_all.drop('income', axis=1, inplace=True)
# industry                        0.937717  删了
df_all.drop('industry', axis=1, inplace=True)
# merriage                        0.932954  删了
df_all.drop('merriage', axis=1, inplace=True)
# name_rec_md5                    0.478054  补充-5
df_all['name_rec_md5'].fillna(-5, inplace=True)
# no_order_md5                    0.474976  补充-100
df_all['no_order_md5'].fillna(-100, inplace=True)
# order_count_id                  0.000000
# order_year1_limit_groupby_id    0.474976  补充0 增加一列 isna
df_all.loc[(df_all.order_year1_limit_groupby_id.notnull()), 'order_year_limit_groupby_id_isna'] = "0"
df_all.loc[(df_all.order_year1_limit_groupby_id.isnull()), 'order_year_limit_groupby_id_isna'] = "1"
df_all['order_year1_limit_groupby_id'].fillna(0, inplace=True)
# order_year2_limit_groupby_id    0.474976  补充0
df_all['order_year2_limit_groupby_id'].fillna(0, inplace=True)
# order_year3_limit_groupby_id    0.474976  补充0
df_all['order_year3_limit_groupby_id'].fillna(0, inplace=True)
# order_year4_limit_groupby_id    0.474976  补充0
df_all['order_year4_limit_groupby_id'].fillna(0, inplace=True)
# overdraft                       0.014539  补充0 增加一列 isna
df_all.loc[(df_all.overdraft.notnull()), 'overdraft_isna'] = "0"
df_all.loc[(df_all.overdraft.isnull()), 'overdraft_isna'] = "1"
df_all['overdraft'].fillna(0, inplace=True)
# phone_auth                      0.025845  补充0
df_all['phone_auth'].fillna(0, inplace=True)
# phone_bankcard                  0.000000
# phone_order_info                0.477233  补充-10
df_all['phone_order_info'].fillna(-10, inplace=True)
# phone_recieve_addr              0.297231  补充-10
df_all['phone_recieve_addr'].fillna(-10, inplace=True)
# qq_bound                        0.037759  补充‘未知’
df_all['qq_bound'].value_counts()  # 2类
df_all['qq_bound'].fillna('未知', inplace=True)
# quota                           0.014539  补充0 增加一列 isna
df_all.loc[(df_all.quota.notnull()), 'quota_isna'] = "0"
df_all.loc[(df_all.quota.isnull()), 'quota_isna'] = "1"
df_all['quota'].fillna(0, inplace=True)
# receiver_recieve_addr           0.297237  补充-10
df_all['receiver_recieve_addr'].fillna(-10, inplace=True)
# region_recieve_addr             0.297231  补充-10
df_all['region_recieve_addr'].fillna(-10, inplace=True)
# sex                             0.030192  补充‘未知’
df_all['sex'].value_counts()  # 3类
df_all['sex'].fillna('未知', inplace=True)
# sts_order                       0.501727  补充-10
df_all['sts_order'].fillna(-10, inplace=True)
# target                          0.280013
# type_pay                        0.478019  补充-5
df_all['type_pay'].fillna(-5, inplace=True)
# user_age                        0.667814  补充-100
df_all['user_age'].fillna(-100, inplace=True)
# wechat_bound                    0.037759  补充‘未知’
df_all['wechat_bound'].value_counts()  # 2类
df_all['wechat_bound'].fillna('未知', inplace=True)

# 相关性检查
cor = df_all.corr()
cor.loc[:, :] = np.tril(cor, k=-1)
cor = cor.stack()
cor[(cor > 0.55) | (cor < -0.55)]
# fix_phone                     addr_id_recieve_addr            0.647604
# name_rec_md5                  addr_id_recieve_addr            0.551576
# fix_phone                       0.553676
# no_order_md5                  name_rec_md5                    0.822679
# order_count_id                no_order_md5                    0.792335
# order_year2_limit_groupby_id  no_order_md5                    0.748671
# order_count_id                  0.922152
# order_year3_limit_groupby_id  order_count_id                  0.576808
# order_year4_limit_groupby_id  order_year3_limit_groupby_id    0.594622
# phone_order_info              addr_id_recieve_addr            0.561852
# name_rec_md5                    0.916045
# no_order_md5                    0.830692
# phone_recieve_addr            addr_id_recieve_addr            0.732286
# fix_phone                       0.659275
# name_rec_md5                    0.575893
# phone_order_info                0.576543
# quota                         overdraft                       0.886667
# receiver_recieve_addr         addr_id_recieve_addr            0.732295
# fix_phone                       0.659288
# name_rec_md5                    0.575904
# phone_order_info                0.576552
# phone_recieve_addr              0.999989
# region_recieve_addr           addr_id_recieve_addr            0.732290
# fix_phone                       0.659282
# name_rec_md5                    0.575896
# phone_order_info                0.576544
# phone_recieve_addr              0.999999
# receiver_recieve_addr           0.999990
# sts_order                     name_rec_md5                    0.850344
# no_order_md5                    0.761096
# phone_order_info                0.861837
# phone_recieve_addr              0.557397
# receiver_recieve_addr           0.557406
# region_recieve_addr             0.557397
# type_pay                      addr_id_recieve_addr            0.565243
# name_rec_md5                    0.903199
# no_order_md5                    0.818616
# phone_order_info                0.909627
# phone_recieve_addr              0.588426
# receiver_recieve_addr           0.588435
# region_recieve_addr             0.588426
# sts_order                       0.932731
# 删除overdraft name_rec_md5
df_all.drop(['overdraft', 'overdraft_isna', 'name_rec_md5'], axis=1, inplace=True)

# one hot
classification = df_all[['account_grade', 'qq_bound', 'sex', 'wechat_bound']]
classification = pd.get_dummies(classification)
df_all = pd.concat([df_all, classification], axis=1, join_axes=[df_all.index])
df_all.drop(['account_grade', 'qq_bound', 'sex', 'wechat_bound'], axis=1, inplace=True)

# 归一化
df_all_scaler = pd.DataFrame(MinMaxScaler().fit_transform(df_all), columns=df_all.columns)

# 切分训练集测试集
train = df_all_scaler[:df_train_row]
test = df_all_scaler[df_train_row:]
x_train, x_test, y_train, y_test = train_test_split(train, feature, test_size=0.3, random_state=123)

# 梯度提升回归数  GBRT 进行拟合
param_grid = {'learning_rate': [0.1],  # 学习速率
              'max_depth': [2],  # 树的最大深度
              'min_samples_split': [50],  # 树中包含样本数
              'n_estimators': [100]  # 树的个数
              }
if __name__ == '__main__':
    est = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(),
                       param_grid=param_grid,
                       n_jobs=5,
                       refit=True)
    d1 = datetime.datetime.now()
    est.fit(x_train, y_train)
    best_param = est.best_params_
    d2 = datetime.datetime.now()
    (d2 - d1).seconds