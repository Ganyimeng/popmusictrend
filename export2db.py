# encoding=utf-8
import mysql.connector

config = {'host': '127.0.0.1',  # 默认127.0.0.1
          'user': 'root',
          'password': '1111',
          'port': 3306,  # 默认即为3306
          'database': 'test',
          'charset': 'utf8'  # 默认即为utf8
          }
try:
    cnn = mysql.connector.connect(**config)
    cursor = cnn.cursor()
    cursor.execute("select * from Persons")
    row = cursor.fetchall()
    print row
    cursor.close()
    cnn.close()
except mysql.connector.Error as e:
    print('connect fails!{}'.format(e))
