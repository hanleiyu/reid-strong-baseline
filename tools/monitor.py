import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import os
import time
import datetime

# 第三方 SMTP 服务
my_host = "smtp.163.com"
my_sender = "15883113154@163.com"
my_user = "601901861@qq.com"  # 收件人的邮箱地址
my_pass = "ASJTYAHSWGPZPIEG"


def mail():
    ret = True
    try:
        msg = MIMEText('ok', 'plain', 'utf-8')
        msg['From'] = formataddr(["FromRunoob", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(["FK", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject'] = "菜鸟教程发送邮件测试"  # 邮件的主题，也可以说是标题

        server = smtplib.SMTP_SSL(my_host, 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret = False
    return ret


def popen_():
    """
    popen模块执行linux命令。返回值是类文件对象，获取结果要采用read()或者readlines()
    :return:
    """
    val = os.popen("nvidia-smi | grep -o '[[:alnum:]]'*MiB").read()
    val = val.replace("MiB\n", " ")
    val = val.rstrip().split(" ")
    flag = 0
    # for i in range(len(val)):
    #     if int(val[i]) < 2000:
    #         flag = 1
    #         break
    if int(val[0]) < 2000:
        flag = 1
    if flag == 0:
        return False
    else:
        return True

while 1 :
    time.sleep(30)
    now = datetime.datetime.now()
    print(now, popen_())
    if popen_():
        ret = mail()
        if ret:
            print("邮件发送成功")
            time.sleep(350)
            mail()
            break
        else:
            print("邮件发送失败")