# 数据库基础 -- 工作向

掌握下面的核心概念

## 1.SQL

什么是SQL？（https://zh.wikipedia.org/wiki/SQL），与数据库交互的DSL，实现对数据的增删查改

## 2. RDBMS (data base manage system) and NRDBMS

有两种类型数据库
- 关系型数据库（使用SQL交互）
- 非关系型数据库（NoSQL，无法使用SQL）

## 3. table and keys

RDBMS 使用一个个table来存储数据，

参考这个教程 https://dunwu.github.io/db-tutorial/pages/b71c9e/#sql-%E7%AE%80%E4%BB%8B

如果想要表格和表格之前产生关联，可以使用 foreign key 映射到 另一张表的（同一张表也可以）primary key

一个table可以设置多个primary key，来定位唯一的row

## 4. 增删查改基本操作

总体来看
增删查改 insert delete update select
判断条件使用 where
逻辑运算 and or
返回结果去重 distinct
聚合函数 count
选定表格 from
集合 union join

## 5. ORM

object relationship mapping
 - 把程序中的对象映射到数据库里面
 - 对于应用开发者，不需要把sql语句嵌入到代码里，直接使用orm框架的API即可！
 - orm会将用户API映射为sql语句给RDBMS执行

## 6. gorm

视频教程 https://www.bilibili.com/video/BV1U7411V78R/?spm_id_from=333.337.search-card.all.click&vd_source=eff5b75cc7d63ea5323561bc3674a62c
中文文档 https://gorm.io/zh_CN/docs/index.html


## QA

1. 为什么table的名字叫 space_instances?
2. 为什么找不到 eci_resource table?
