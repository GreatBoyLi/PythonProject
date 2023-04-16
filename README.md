# PythonProject
My own Python project PostGraduate

# Linux and Python
使用CentOs运行Python

# CentOS新增用户
useradd -m -s /bin/bash code 
 -m 创建 home 文件夹

passwd code  
 修改code用户的密码，需使用root用户

# 安装 miniconda 
wget <复制的链接地址>
sh ./ Miniconda3-latest-Linux-x86_64.sh
conda 命令在 miniconda目录下的bin目录下

# 安装Git
sudo yum update
sudo yum install git
git --version
如果没有安装 yum包管理器
sudo dnf install yum 或者 sudo apt-get install yum


# 永久修改pip镜像源
Windows：新建 pip.ini 文件存放在 User\pip\pip.ini
Linux：新建pip.conf文件存放在 ~/.pip/pip.conf

[global]
trusted-host=mirrors.aliyun.com
index-url=http://mirrors.aliyun.com/pypi/simple/


